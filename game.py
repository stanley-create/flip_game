import numpy as np
import copy
import time

class FlipGame:
    def __init__(self):
        # 0=空, 1=正(一般), -1=反(一般), 2=正(鬼牌), -2=反(鬼牌)
        self.board = np.zeros((4, 4), dtype=int)
        self.current_player = 1
        self.scores = {1: [0, 0], 2: [0, 0]}  # [條數, 鬼牌數]
        self.game_over = False
        self.winner = None

    def display(self):
        symbols = {0: ' . ', 1: ' ○ ', -1: ' ● ', 2: ' ☼ ', -2: ' ☀ '}
        print("\n   0  1  2  3 (X)")
        for y in range(4):
            row_str = f"{y} "
            for x in range(4):
                row_str += symbols[self.board[y, x]]
            print(row_str)
        print(f"玩家 1 得分: {self.scores[1][0]} 條, {self.scores[1][1]} 鬼牌")
        print(f"玩家 2 得分: {self.scores[2][0]} 條, {self.scores[2][1]} 鬼牌\n")

    def get_valid_moves(self):
        valid_moves = []
        for y in range(4):
            for x in range(4):
                if self.board[y, x] == 0:
                    valid_moves.extend([(y, x, 1), (y, x, -1)])
        return valid_moves

    def step(self, y, x, side, is_joker=False):
        if self.board[y, x] != 0:
            return False, "該位置已有卡牌"

        value = (2 if is_joker else 1) * side
        self.board[y, x] = value

        # 十字翻轉
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < 4 and 0 <= nx < 4 and self.board[ny, nx] != 0:
                self.board[ny, nx] *= -1

        self.check_and_collect_lines()
        self.check_winner()

        if not self.game_over:
            self.current_player = 3 - self.current_player
        return True, "成功"

    def check_and_collect_lines(self):
        lines_to_collect = []
        # 1. 橫列
        for y in range(4):
            row = self.board[y, :]
            if np.all(row > 0) or np.all(row < 0):
                lines_to_collect.append(('row', y, row.copy()))
        # 2. 直行
        for x in range(4):
            col = self.board[:, x]
            if np.all(col > 0) or np.all(col < 0):
                lines_to_collect.append(('col', x, col.copy()))
        # 3. 主對角線
        diag1 = np.array([self.board[i, i] for i in range(4)])
        if np.all(diag1 > 0) or np.all(diag1 < 0):
            lines_to_collect.append(('diag1', None, diag1))
        # 4. 副對角線
        diag2 = np.array([self.board[i, 3 - i] for i in range(4)])
        if np.all(diag2 > 0) or np.all(diag2 < 0):
            lines_to_collect.append(('diag2', None, diag2))

        if not lines_to_collect:
            return

        # 核心自動收線
        line_type, idx, cells = lines_to_collect[0]
        jokers_count = int(np.sum(np.abs(cells) == 2))
        self.scores[self.current_player][0] += 1
        self.scores[self.current_player][1] += jokers_count

        if line_type == 'row': self.board[idx, :] = 0
        elif line_type == 'col': self.board[:, idx] = 0
        elif line_type == 'diag1':
            for i in range(4): self.board[i, i] = 0
        elif line_type == 'diag2':
            for i in range(4): self.board[i, 3 - i] = 0

    def check_winner(self):
        for p in [1, 2]:
            lines, jokers = self.scores[p]
            opponent_lines = self.scores[3 - p][0]
            if jokers >= 2 or (jokers == 1 and lines >= opponent_lines + 2):
                self.game_over = True
                self.winner = p

# =====================================================================
# 🧠 評估函數 (Heuristic Evaluation) —— 演算法的靈魂
# =====================================================================
def evaluate_board(game, player):
    """評估目前盤面對於指定 player 的有利程度"""
    if game.game_over:
        return 9999 if game.winner == player else -9999

    opp = 3 - player
    # 基礎分數 = (我的條數 - 敵方條數) * 100 + (我的鬼牌 - 敵方鬼牌) * 300
    score = (game.scores[player][0] - game.scores[opp][0]) * 100 \
          + (game.scores[player][1] - game.scores[opp][1]) * 300

    # 潛力分：檢查盤面上快要連成線的「聽牌」狀態
    for y in range(4):
        # 橫列潛力
        row = game.board[y, :]
        score += np.sum(row == (1 if player == 1 else -1)) * 5
    return score

# =====================================================================
# 🚀 終極 Minimax 核心 (帶 Alpha-Beta 剪枝)
# =====================================================================
def minimax(game_state, depth, alpha, beta, maximizing_player, target_player):
    if depth == 0 or game_state.game_over:
        return evaluate_board(game_state, target_player), None

    valid_moves = game_state.get_valid_moves()
    if not valid_moves:
        return evaluate_board(game_state, target_player), None

    # 🔥 動作排序優化 (Move Ordering)：優先評估可能有高分的步，提升剪枝效率
    # 這能讓深度 5 的計算速度加快數倍！
    valid_moves.sort(key=lambda m: (abs(m[0]-1.5) + abs(m[1]-1.5))) 

    best_move = None

    if maximizing_player:
        max_eval = -float('inf')
        for y, x, side in valid_moves:
            # 複製虛擬盤面進行預判
            sim_game = copy.deepcopy(game_state)
            sim_game.step(y, x, side)
            
            # 往下搜尋（注意切換成對手的視角）
            is_next_max = (sim_game.current_player == target_player)
            ev, _ = minimax(sim_game, depth - 1, alpha, beta, is_next_max, target_player)
            
            if ev > max_eval:
                max_eval = ev
                best_move = (y, x, side)
            alpha = max(alpha, ev)
            if beta <= alpha: # 💥 Beta 剪枝：後面不用算了，直接砍掉
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for y, x, side in valid_moves:
            sim_game = copy.deepcopy(game_state)
            sim_game.step(y, x, side)
            
            is_next_max = (sim_game.current_player == target_player)
            ev, _ = minimax(sim_game, depth - 1, alpha, beta, is_next_max, target_player)
            
            if ev < min_eval:
                min_eval = ev
                best_move = (y, x, side)
            beta = min(beta, ev)
            if beta <= alpha: # 💥 Alpha 剪枝
                break
        return min_eval, best_move

def get_god_hint(game, depth=5):
    """調用深度 5 的 Minimax 獲取神級提示"""
    current_p = game.current_player
    _, move = minimax(game, depth, -float('inf'), float('inf'), True, current_p)
    return move

# =====================================================================
# 遊戲主控制流
# =====================================================================
def play_game():
    game = FlipGame()
    print("====================================")
    print("=== 4x4 正反棋：深度 5 終極 AI 版 ===")
    print("====================================")
    mode = input("選擇模式 (A: 雙人本地, B: 神級AI最佳解提示, C: 人機對戰(VS 深度5 AI)): ").upper()

    print("\n【準備階段】玩家 1 請放置兩張 Joker 鬼牌")
    for i in range(2):
        game.display()
        while True:
            try:
                inputs = input(f"請輸入第 {i+1} 張 Joker 的座標與面向 (y x side): ").split()
                y, x, side = map(int, inputs)
                success, msg = game.step(y, x, side, is_joker=True)
                if success: break
                print(f"錯誤: {msg}")
            except (ValueError, IndexError):
                print("輸入格式錯誤。")
    
    game.current_player = 2

    while not game.game_over:
        game.display()
        print(f"--- 輪到玩家 {game.current_player} ---")
        
        # 模式 B：神級提示
        if mode == 'B':
            t0 = time.time()
            print("🔮 神級 AI 正在通靈預判 5 步棋...")
            hint = get_god_hint(game, depth=5)
            print(f"💡 [AI 最佳提示] (耗時 {time.time()-t0:.2f}秒) -> 座標: ({hint[0]}, {hint[1]}) 面向: {hint[2]}")

        # 模式 C：人機對戰
        if mode == 'C' and game.current_player == 2:
            print("🤖 深度 5 AI 計算中...")
            t0 = time.time()
            move = get_god_hint(game, depth=5)
            if move:
                print(f"🤖 AI 落子 (耗時 {time.time()-t0:.2f}秒) -> 座標: ({move[0]}, {move[1]}) 面向: {move[2]}")
                game.step(move[0], move[1], move[2])
        else:
            while True:
                try:
                    inputs = input("請輸入卡牌座標與面向 (y x side): ").split()
                    y, x, side = map(int, inputs)
                    success, msg = game.step(y, x, side)
                    if success: break
                    print(f"錯誤: {msg}")
                except (ValueError, IndexError):
                    print("輸入格式錯誤。")
                
    game.display()
    print(f"🎉 遊戲結束！恭喜 玩家 {game.winner} 獲得最終勝利！")

if __name__ == "__main__":
    play_game()