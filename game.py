import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import os
import time
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# 1. 偵測環境與設定路徑
IS_COLAB = 'COLAB_GPU' in os.environ or os.path.exists('/content')

if IS_COLAB:
    print("🤖 偵測到 Colab 環境，掛載 Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    MODEL_DIR = "/content/drive/MyDrive/AI"
else:
    print("💻 偵測到本地/伺服器環境，使用本地資料夾 ./models")
    MODEL_DIR = os.path.join(os.getcwd(), "models")

# 2. 自動建立目錄
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"✅ 已建立儲存目錄: {MODEL_DIR}")

# 3. 定義路徑變數
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_blind_v2_model")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_ppo_blind_v2_model")
RECORD_TXT_PATH = os.path.join(MODEL_DIR, "best_blind_v2_record.txt")
GAS_URL = "https://script.google.com/macros/s/AKfycbwYE8yW0Vx3eR9W_LIadIMI9TVI0olXAT10YkCec6ZPdbken1kzX4k30CBEZ4Fm-d4o/exec"

# =====================================================================
# 2. 核心遊戲邏輯（真實棋盤，完美對接 Web 網頁端）
# =====================================================================
class FlipGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.game_over = False
        self.current_player = 1
        self.scores = {1: [0, 0], 2: [0, 0]}  # 格式：{ 1: [連線數, 鬼牌數], 2: [連線數, 鬼牌數] }

    def step(self, y, x, side, is_joker=False):
        if self.board[y, x] != 0:
            return False, "該位置已有棋子！"

        if is_joker:
            # 佈置鬼牌：正向為 2，反向為 -2
            self.board[y, x] = 2 if side == 1 else -2
            self.update_scores()
            return True, "成功佈置鬼牌 🃏"

        # 常規落子
        self.board[y, x] = side
        # 翻轉十字鄰近棋子
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < 4 and 0 <= nx < 4 and self.board[ny, nx] != 0:
                self.board[ny, nx] *= -1

        self.update_scores()
        # 轉換玩家回合
        self.current_player = 2 if self.current_player == 1 else 1
        return True, "落子成功"

    def update_scores(self):
        """ 核心分數同步：計算連線數與鬼牌保管數 """
        p1_lines = self.count_lines(1)
        p2_lines = self.count_lines(-1)
        p1_jokers = np.sum(self.board == 2)
        p2_jokers = np.sum(self.board == -2)
        self.scores[1] = [p1_lines, p1_jokers]
        self.scores[2] = [p2_lines, p2_jokers]

    def count_lines(self, player_side):
        """ 計算 4x4 棋盤上的直線、橫線與對角線連線數 """
        lines = 0
        for y in range(4):
            if np.all(self.board[y, :] == player_side): lines += 1
        for x in range(4):
            if np.all(self.board[:, x] == player_side): lines += 1
        if np.all(np.diag(self.board) == player_side): lines += 1
        if np.all(np.diag(np.fliplr(self.board)) == player_side): lines += 1
        return lines

def check_custom_game_over(board_matrix):
    has_joker = np.any((board_matrix == 2) | (board_matrix == -2))
    is_full = not np.any(board_matrix == 0)
    return (not has_joker) or is_full

# =====================================================================
# 3. 強化學習環境封裝（純淨盲棋面罩版 🎭 - 移除危險投降機制）
# =====================================================================
class FlipGameEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = FlipGame()
        self.action_space = spaces.Discrete(32)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(4, 4), dtype=np.int32)

    def _get_masked_obs(self):
        """ 🎭 盲棋面罩：將所有 -2 (鬼牌背面) 偽裝成 -1 """
        obs = self.game.board.copy()
        obs[obs == -2] = -1
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()

        joker_positions = np.random.choice(16, size=2, replace=False)
        for pos in joker_positions:
            y, x = pos // 4, pos % 4
            side = np.random.choice([2, -2])
            self.game.board[y, x] = side

        return self._get_masked_obs(), {}

    def step(self, action):
        y, x, side = action // 8, (action % 8) // 2, 1 if action % 2 == 0 else -1
        
        # 修正環境相容性：解包 step 的新雙回傳值
        valid, _ = self.game.step(y, x, side)
        
        if not valid:
            return self._get_masked_obs(), -100, True, False, {}
            
        game_over = check_custom_game_over(self.game.board)
        
        if game_over:
            p1 = np.sum(self.game.board > 0)
            p2 = np.sum(self.game.board < 0)
            if p1 > p2: reward = 100
            elif p1 < p2: reward = -100
            else: reward = 0
        else:
            reward = -0.1

        return self._get_masked_obs(), reward, game_over, False, {}

# =====================================================================
# 🚨 終極安全防護牆：使用標準 python 魔術防火牆鎖死訓練
# =====================================================================
# 運作原理：
# 1. 當網頁 app.py 透過 "import" 引用此檔案時，以下所有代碼會被 100% 完美屏蔽，絕不執行！
# 2. 未來如果你想在電腦或 Colab 執行訓練，直接運行 `python game.py` 依然能正常啟動訓練！
if __name__ == "__main__":
    env = FlipGameEnv()
    env = Monitor(env)

    if os.path.exists(MODEL_PATH + ".zip"):
        print("🤖 偵測到 v2 盲棋模型，載入中繼續訓練...")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("🆕 建立全新的 v2 PPO 模型 (純淨盲棋公平版)...")
        model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.02)

    if os.path.exists(RECORD_TXT_PATH):
        try:
            with open(RECORD_TXT_PATH, "r") as f:
                lines = f.read().splitlines()
                best_mean_reward = float(lines[0])
                best_win_rate = float(lines[1])
            print(f"💾 成功讀取 v2 最高紀錄！歷史最高勝率: {best_win_rate:.2f}%")
        except Exception as e:
            print(f"⚠️ 讀取紀錄檔失敗，重新初始化: {e}")
            best_mean_reward = -np.inf
            best_win_rate = -1.0
    else:
        best_mean_reward = -np.inf
        best_win_rate = -1.0

    print("🚀 開始健康訓練（移除投降干擾，AI 將逐步學會不犯規與獲勝）...")
    for i in range(1000):
        model.learn(total_timesteps=5000, reset_num_timesteps=False)
        model.save(MODEL_PATH)
        
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)
        current_win_rate = (mean_reward + 100) / 2 

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_win_rate = current_win_rate
            model.save(BEST_MODEL_PATH)
            with open(RECORD_TXT_PATH, "w") as f:
                f.write(f"{best_mean_reward}\n{best_win_rate}")
            print(f"🎉 突破紀錄！新最佳勝率: {best_win_rate:.2f}%，已安全存檔！")
        else:
            print(f" 🚩 本輪評估勝率: {current_win_rate:.2f}% (未突破歷史最高 {best_win_rate:.2f}%)")

        total_steps = int(model.num_timesteps)
        payload = {
            "steps": total_steps,
            "avg_reward": float(round(mean_reward, 2)),
            "win_rate": float(round(current_win_rate, 2)),
            "best_win_rate": float(round(best_win_rate, 2))
        }

        try:
            response = requests.post(GAS_URL, json=payload, timeout=15)
            response.raise_for_status()
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{total_steps} 步] 數據已同步")
        except Exception as e:
            print(f"⚠️ 同步暫時失敗: {e}")

        time.sleep(5)