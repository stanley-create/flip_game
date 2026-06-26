import streamlit as st
import numpy as np
import copy
import time
import os
import json
import random

# 💡 導入強化學習與 Firebase 套件
from stable_baselines3 import PPO
import firebase_admin
from firebase_admin import credentials, db

# 載入核心邏輯
from game import FlipGame

# =====================================================================
# 🧰 載入外部 CSS 檔案 與 初始化設定
# =====================================================================
st.set_page_config(page_title="FlipGame 戰術儀表板", layout="wide")

if os.path.exists("styles.css"):
    with open("styles.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

DIFFICULTY_MODELS = {
    "🌱 簡單模式": "best_ppo_blind_v2_model_easy",
    "🔥 中等難度 (94%)": "best_ppo_blind_v2_model_94",
    "💀 地獄模式 (X%)": "best_ppo_blind_v2_model_hell"
}

LEADERBOARD_FILE = "leaderboard.json"

# =====================================================================
# 🔥 Firebase 即時雲端初始化
# =====================================================================
if not firebase_admin._apps:
    try:
        # 從 Streamlit Secrets 讀取憑證
        cred_dict = dict(st.secrets["firebase_service_account"])
        cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
        
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred, {
            'databaseURL': st.secrets["firebase_database_url"]
        })
    except Exception as e:
        st.error(f"⚠️ Firebase 未連線或 Secrets 未設定：{e} (線上對戰模式將無法使用)")

# =====================================================================
# 🏆 本地排行榜儲存核心函數
# =====================================================================
def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"🔵 玩家 (藍方)": 0, "🔴 AI / 紅方": 0, "🤝 平手": 0, "🌐 線上對抗賽": 0}
    return {"🔵 玩家 (藍方)": 0, "🔴 AI / 紅方": 0, "🤝 平手": 0, "🌐 線上對抗賽": 0}

def save_leaderboard(winner_code, is_online=False):
    board_data = load_leaderboard()
    if is_online:
        board_data["🌐 線上對抗賽"] += 1
    else:
        if winner_code == 1:
            board_data["🔵 玩家 (藍方)"] += 1
        elif winner_code == 2:
            board_data["🔴 AI / 紅方"] += 1
        else:
            board_data["🤝 平手"] += 1
    with open(LEADERBOARD_FILE, "w", encoding="utf-8") as f:
        json.dump(board_data, f, ensure_ascii=False, indent=4)

# =====================================================================
# ⚙️ 初始化 Session State
# =====================================================================
if 'game' not in st.session_state:
    st.session_state.game = FlipGame()
    st.session_state.phase = "PREPARE"
    st.session_state.jokers_placed = 0
    st.session_state.hint_move = None
    st.session_state.prev_board = np.zeros((4, 4), dtype=int)
    st.session_state.go_first = "先手 (我布置鬼牌)"
    st.session_state.current_model_path = ""
    
    # 歷史紀錄與本地日誌
    st.session_state.game_history = []
    st.session_state.log_history = []
    st.session_state.move_log = ["🎬 賽局開始，進入鬼牌佈置階段..."]
    st.session_state.leaderboard_updated = False

# 🆕 聯網狀態初始化
if "user_id" not in st.session_state:
    st.session_state.user_id = f"User_{random.randint(1000, 9999)}"
if "net_mode" not in st.session_state:
    st.session_state.net_mode = "LOBBY" # LOBBY, MATCHING, ROOM
if "room_id" not in st.session_state:
    st.session_state.room_id = None
if "my_role" not in st.session_state:
    st.session_state.my_role = None  # PLAYER_1 (先手藍), PLAYER_2 (後手紅)
if "match_start_time" not in st.session_state:
    st.session_state.match_start_time = None

game = st.session_state.game

# =====================================================================
# 🔄 遊戲核心動作封裝 (整合本地紀錄與 Firebase 同步)
# =====================================================================
def get_coord_name(y, x):
    return f"{['A', 'B', 'C', 'D'][y]}{x+1}"

def check_custom_game_over(board_matrix):
    has_joker = np.any((board_matrix == 2) | (board_matrix == -2))
    is_full = not np.any(board_matrix == 0)
    return (not has_joker) or is_full

def execute_game_step(y, x, side, is_joker=False):
    """ 萬能落子控制：自動記錄日誌，若為線上模式則自動同步到 Firebase """
    is_online = (st.session_state.play_mode == "🌐 全球線上對戰 (Firebase)")
    
    if not is_online:
        # 單機模式：備份快照供悔棋使用
        st.session_state.game_history.append(copy.deepcopy(st.session_state.game))
        st.session_state.log_history.append(list(st.session_state.move_log))
    
    # 取得當前下棋標籤
    if is_online:
        p_label = "🔵 先手藍方" if game.current_player == 1 else "🔴 後手紅方"
    else:
        p_label = "🔵 藍方" if game.current_player == 1 else "🔴 紅方/AI"
        
    success, msg = game.step(y, x, side, is_joker=is_joker)
    
    if success:
        st.session_state.hint_move = None
        coord = get_coord_name(y, x)
        if is_joker:
            side_label = "🟦 藍色正面" if side == 1 else "🟥 灰色隱藏"
            st.session_state.move_log.append(f"{p_label} 布置了 {side_label} 鬼牌 🃏 於 {coord}")
        else:
            side_label = "🟦 正面" if side == 1 else "🟥 反面"
            st.session_state.move_log.append(f"{p_label} 落子於 {coord} ({side_label})")
            
        # 🌐 線上模式：立刻把落子結果推播上雲端
        if is_online and st.session_state.room_id:
            if check_custom_game_over(game.board):
                st.session_state.phase = "OVER"
            
            db.reference(f"rooms/{st.session_state.room_id}").update({
                "board": game.board.tolist(),
                "current_player": game.current_player,
                "status": st.session_state.phase,
                "scores": {1: list(game.scores[1]), 2: list(game.scores[2])},
                "move_log": st.session_state.move_log
            })
    else:
        if not is_online:
            st.session_state.game_history.pop()
            st.session_state.log_history.pop()
    return success, msg

def undo_last_move():
    """ 悔棋控制引擎 (僅限單機模式) """
    if not st.session_state.game_history:
        st.toast("⚠️ 沒有更早的歷史紀錄可以悔棋了！")
        return

    is_vs_ai = st.session_state.play_mode == "人機對戰 (VS PPO強化學習大腦)"
    
    if is_vs_ai and game.current_player == 1 and len(st.session_state.game_history) >= 2:
        st.session_state.game_history.pop()  
        st.session_state.log_history.pop()
        st.session_state.game = st.session_state.game_history.pop()
        st.session_state.move_log = st.session_state.log_history.pop()
    else:
        st.session_state.game = st.session_state.game_history.pop()
        st.session_state.move_log = st.session_state.log_history.pop()

    st.session_state.hint_move = None
    if st.session_state.phase == "OVER":
        st.session_state.phase = "PLAY"
        st.session_state.leaderboard_updated = False
        
    jokers = np.sum(np.abs(st.session_state.game.board) == 2)
    st.session_state.jokers_placed = jokers
    if jokers < 2:
        st.session_state.phase = "PREPARE"
        
    st.toast("⏪ 悔棋成功！時空已倒流。")
    st.rerun()

# =====================================================================
# 🧠 PPO 強化學習大腦載入與決策
# =====================================================================
def load_selected_model(model_path):
    if st.session_state.get('current_model_path') == model_path and 'rl_model' in st.session_state:
        return st.session_state.rl_model
    if os.path.exists(model_path + ".zip"):
        try:
            model = PPO.load(model_path)
            st.session_state.current_model_path = model_path
            st.session_state.rl_model = model
            return model
        except Exception as e:
            st.sidebar.error(f"❌ 模型損壞: {str(e)}")
            return None
    return None

def get_rl_ai_move(game_instance, model):
    if model is None:
        empty_slots = np.argwhere(game_instance.board == 0)
        idx = np.random.choice(len(empty_slots))
        return empty_slots[idx][0], empty_slots[idx][1], np.random.choice([1, -1])
    
    masked_obs = game_instance.board.copy()
    masked_obs[masked_obs == -2] = -1
    masked_obs = masked_obs.astype(np.float32)
    
    action, _ = model.predict(masked_obs, deterministic=True)
    action = int(action)
    y = action // 8
    x = (action % 8) // 2
    side = 1 if action % 2 == 0 else -1
    return y, x, side

# =====================================================================
# 🌐 Firebase 線上大廳與匹配邏輯控制
# =====================================================================
def start_firebase_matchmaking():
    st.session_state.net_mode = "MATCHING"
    st.session_state.match_start_time = time.time()
    db.reference("matchmaking_queue").child(st.session_state.user_id).set({
        "timestamp": time.time(),
        "status": "WAITING"
    })
    st.toast("🚀 已進入全球匹配隊列...")

def cancel_firebase_matchmaking():
    db.reference(f"matchmaking_queue/{st.session_state.user_id}").delete()
    st.session_state.net_mode = "LOBBY"
    st.toast("❌ 已退出匹配")

def check_matchmaking_loop():
    user_id = st.session_state.user_id
    my_status = db.reference(f"matchmaking_queue/{user_id}").get()
    if not my_status: return
    
    if my_status.get("status") == "MATCHED":
        st.session_state.room_id = my_status.get("room_id")
        st.session_state.my_role = "PLAYER_2"
        st.session_state.net_mode = "ROOM"
        db.reference(f"matchmaking_queue/{user_id}").delete()
        st.balloons()
        st.rerun()
        
    queue = db.reference("matchmaking_queue").get()
    if queue:
        for opp_id, data in queue.items():
            if opp_id != user_id and data.get("status") == "WAITING":
                room_code = str(random.randint(1000, 9999))
                room_ref = db.reference(f"rooms/{room_code}")
                room_ref.set({
                    "player_1": user_id,
                    "player_2": opp_id,
                    "status": "PREPARE",
                    "current_player": 1,
                    "board": [[0]*4 for _ in range(4)],
                    "scores": {"1": [0, 0], "2": [0, 0]},
                    "move_log": ["🎬 網路快速匹配成功！請 先手藍方 佈置第 1 張鬼牌..."]
                })
                db.reference(f"matchmaking_queue/{opp_id}").update({
                    "status": "MATCHED",
                    "room_id": room_code
                })
                st.session_state.room_id = room_code
                st.session_state.my_role = "PLAYER_1"
                st.session_state.net_mode = "ROOM"
                db.reference(f"matchmaking_queue/{user_id}").delete()
                st.balloons()
                st.rerun()

def create_firebase_private_room():
    room_code = str(random.randint(1000, 9999))
    db.reference(f"rooms/{room_code}").set({
        "player_1": st.session_state.user_id,
        "player_2": None,
        "status": "WAITING_PLAYER",
        "current_player": 1,
        "board": [[0]*4 for _ in range(4)],
        "scores": {"1": [0, 0], "2": [0, 0]},
        "move_log": [f"🏠 私人房間 {room_code} 已創建，等待好友輸入房號加入..."]
    })
    st.session_state.room_id = room_code
    st.session_state.my_role = "PLAYER_1"
    st.session_state.net_mode = "ROOM"

def join_firebase_private_room(room_code):
    room_ref = db.reference(f"rooms/{room_code}").get()
    if room_ref:
        if room_ref.get("player_2") is None:
            db.reference(f"rooms/{room_code}").update({
                "player_2": st.session_state.user_id,
                "status": "PREPARE",
                "move_log": room_ref.get("move_log", []) + ["🎉 玩家 2 已輸入房號成功連線，開始佈置鬼牌！"]
            })
            st.session_state.room_id = room_code
            st.session_state.my_role = "PLAYER_2"
            st.session_state.net_mode = "ROOM"
            st.balloons()
            st.rerun()
        else: st.error("⚠️ 該房間人數已滿！")
    else: st.error("⚠️ 找不到該房號，請重新確認。")

# =====================================================================
# ⚙️ 側邊欄 (控制中心)
# =====================================================================
with st.sidebar:
    st.title("😈 地獄盲棋控制")
    st.write(f"🌐 我的聯網 ID: `{st.session_state.user_id}`")
    st.write("---")
    
    selected_diff = st.selectbox("🧠 選擇 AI 大腦難度等級", list(DIFFICULTY_MODELS.keys()), index=1)
    target_path = DIFFICULTY_MODELS[selected_diff]
    current_brain = load_selected_model(target_path)
    
    if current_brain is not None:
        st.success(f"🟢 {selected_diff} 大腦已成功連線！")
    else:
        st.error(f"⚠️ 找不到 `{target_path}.zip`！")
        
    st.write("---")
    st.markdown("### 🛠️ 戰局干涉")
    
    # 聯網模式下鎖定悔棋
    is_now_online = (st.session_state.get("play_mode") == "🌐 全球線上對戰 (Firebase)")
    if st.button("⏪ 執行全宇宙悔棋", type="secondary", use_container_width=True, disabled=(len(st.session_state.game_history) == 0 or is_now_online)):
        undo_last_move()
    if is_now_online:
        st.caption("🔒 線上聯網對戰模式下無法使用悔棋。")
        
    if st.button("🔄 重置整個賽局 (單機)", type="primary", use_container_width=True):
        st.session_state.game = FlipGame()
        st.session_state.phase = "PREPARE"
        st.session_state.jokers_placed = 0
        st.session_state.hint_move = None
        st.session_state.game_history = []
        st.session_state.log_history = []
        st.session_state.move_log = ["🎬 賽局重置，重新進入鬼牌佈置階段..."]
        st.session_state.leaderboard_updated = False
        st.rerun()

    st.write("---")
    st.markdown("### 🏆 歷代累積排行榜")
    scores_data = load_leaderboard()
    for name, win_count in scores_data.items():
        st.write(f"🏆 **{name}** : `{win_count} 勝`")

# 頂部對弈核心模式切換
st.title("🧱 賽博正反棋 : TILE MATRIX")
selected_mode = st.selectbox(
    "🤖 核心對弈模式切換", 
    ["人機對戰 (VS PPO強化學習大腦)", "神級AI最佳解提示", "雙人本地對戰", "🌐 全球線上對戰 (Firebase)"],
    key="play_mode"
)
st.write("---")

# =====================================================================
# 🌐 Firebase 大廳 UI 介面切割切換
# =====================================================================
show_game_board = True

if selected_mode == "🌐 全球線上對戰 (Firebase)":
    if st.session_state.net_mode == "LOBBY":
        show_game_board = False
        col_match, col_room = st.columns(2, gap="large")
        with col_match:
            st.markdown("### ⚔️ 全球天梯快速匹配")
            st.write("透過 Firebase 雲端即時媒合線上的真實棋手。")
            if st.button("🔥 開始即時匹配", type="primary", use_container_width=True):
                start_firebase_matchmaking()
                st.rerun()
        with col_room:
            st.markdown("### 🏠 私人好友對戰")
            if st.button("➕ 創建新房間 (產生房號)", use_container_width=True):
                create_firebase_private_room()
                st.rerun()
            st.write("")
            input_code = st.text_input("🔑 輸入 4 位數房號進房：", placeholder="例如：8888")
            if st.button("🚪 驗證並加入房間", use_container_width=True):
                if input_code: join_firebase_private_room(input_code)

    elif st.session_state.net_mode == "MATCHING":
        show_game_board = False
        st.markdown("<div style='text-align: center; padding: 20px;'>", unsafe_allow_html=True)
        st.subheader("🔍 正在透過 Firebase 搜尋宇宙中的對手...")
        elapsed = int(time.time() - st.session_state.match_start_time)
        estimated_time = 15
        st.progress(min(elapsed / estimated_time, 1.0))
        
        c1, c2 = st.columns(2)
        c1.metric("⏳ 已等待時間", f"{elapsed} 秒")
        c2.metric("⏱️ 預計配對時間", f"{estimated_time} 秒")
        
        if st.button("❌ 取消匹配並返回大廳", use_container_width=True):
            cancel_firebase_matchmaking()
            st.rerun()
            
        check_matchmaking_loop()
        time.sleep(1)
        st.rerun()

    elif st.session_state.net_mode == "ROOM":
        # 進入房間，先拉取雲端即時狀態
        room_data = db.reference(f"rooms/{st.session_state.room_id}").get()
        if not room_data:
            st.error("房間已被解散或不存在。")
            st.session_state.net_mode = "LOBBY"
            time.sleep(1)
            st.rerun()
            
        if room_data.get("status") == "WAITING_PLAYER":
            show_game_board = False
            st.warning(f"🏠 房間建立成功！請將房號告訴朋友：【 {st.session_state.room_id} 】")
            st.info("⏳ 正在等待好友連線進房...")
            if st.button("🚪 關閉房間並返回大廳", type="secondary"):
                db.reference(f"rooms/{st.session_state.room_id}").delete()
                st.session_state.net_mode = "LOBBY"
                st.rerun()
            time.sleep(1)
            st.rerun()
        else:
            # 兩人都就位，完全將雲端數據覆蓋本地變數，達成高強度同步
            game.board = np.array(room_data.get("board"))
            game.current_player = room_data.get("current_player")
            st.session_state.phase = room_data.get("status")
            st.session_state.move_log = room_data.get("move_log", [])
            # 轉換 Firebase 字典分數格式
            fb_scores = room_data.get("scores", {})
            game.scores[1] = fb_scores.get("1", [0, 0])
            game.scores[2] = fb_scores.get("2", [0, 0])
            
            # 計算目前的鬼牌佈置數
            st.session_state.jokers_placed = np.sum(np.abs(game.board) == 2)

# =====================================================================
# 📐 三欄式戰術主版面 (渲染棋盤與狀態)
# =====================================================================
if show_game_board:
    col_left, col_center, col_right = st.columns([1, 1.8, 1.2], gap="large")

    # ---------------------------------------------------------------------
    # ⬅️ 左側欄：動作選擇與戰術日誌
    # ---------------------------------------------------------------------
    with col_left:
        st.markdown("### 🎛️ 戰術動作")
        
        if st.session_state.phase == "PREPARE":
            if selected_mode != "🌐 全球線上對戰 (Firebase)":
                st.write("⏱️ **對局順序設定**")
                prev_go_first = st.session_state.go_first
                st.session_state.go_first = st.radio("請選擇你的順位：", ["先手 (我布置鬼牌)", "後手 (AI 布置鬼牌)"], label_visibility="collapsed")
                if prev_go_first != st.session_state.go_first and st.session_state.jokers_placed == 0:
                    st.rerun()
                st.write("---")
                
            if selected_mode == "🌐 全球線上對戰 (Firebase)":
                st.write("🔧 **設定鬼牌初始面向**")
                side_to_place = st.radio("選擇面向：", [1, -1], format_func=lambda x: "🟦 藍色 (正面方塊)" if x==1 else "🟥 灰色 (隱藏方塊)")
            else:
                if st.session_state.go_first == "先手 (我布置鬼牌)":
                    st.write("🔧 **設定鬼牌初始面向**")
                    side_to_place = st.radio("選擇面向：", [1, -1], format_func=lambda x: "🟦 藍色 (正面方塊)" if x==1 else "🟥 灰色 (隱藏方塊)")
                else:
                    side_to_place = -1
                
        elif st.session_state.phase == "PLAY":
            if selected_mode != "🌐 全球線上對戰 (Firebase)":
                st.markdown(f"⏱️ 順位：`{st.session_state.go_first}`")
                st.write("---")
            st.write("🎨 **選擇本次落子面向**")
            # 判斷是否非 AI 回合
            if not (selected_mode == "人機對戰 (VS PPO強化學習大腦)" and game.current_player == 2):
                side_to_place = st.radio("選擇面向：", [1, -1], format_func=lambda x: "🟦 淺藍色 (正面)" if x==1 else "🟥 灰紅色 (反面)")
            else:
                side_to_place = 1
        else:
            st.success("🏁 賽局已結束")
            side_to_place = 1

        st.write("---")
        st.markdown("### 📜 即時戰術日誌")
        log_html = f"""
        <div style="background-color: #111827; padding: 12px; border-radius: 6px; max-height: 250px; overflow-y: auto; font-family: monospace; font-size: 13px; color: #10B981; border: 1px solid #1F2937;">
            {"<br>".join(st.session_state.move_log[::-1])}
        </div>
        """
        st.markdown(log_html, unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # 🎯 中央欄：核心方塊棋盤
    # ---------------------------------------------------------------------
    with col_center:
        cols_top = st.columns([0.5, 1, 1, 1, 1])
        for x in range(4):
            cols_top[x+1].markdown(f"<div class='axis-label'>{x+1}</div>", unsafe_allow_html=True)

        # 判斷聯網模式下目前是否輪到我的回合
        is_my_turn_online = True
        if selected_mode == "🌐 全球線上對戰 (Firebase)":
            is_my_turn_online = (game.current_player == 1 and st.session_state.my_role == "PLAYER_1") or \
                                 (game.current_player == 2 and st.session_state.my_role == "PLAYER_2")

        for y in range(4):
            cols = st.columns([0.5, 1, 1, 1, 1])
            cols[0].markdown(f"<div class='axis-label'>{['A', 'B', 'C', 'D'][y]}</div>", unsafe_allow_html=True)
            
            for x in range(4):
                val = game.board[y, x]
                if val == 0: button_text = "⬛"      
                elif val == 1: button_text = "🟦"      
                elif val == -1: button_text = "🟥"      
                elif val == 2: button_text = "🃏 J"    
                elif val == -2: button_text = "🟥 " # 鬼牌背面
                    
                if selected_mode == "神級AI最佳解提示" and st.session_state.hint_move and (y, x) == (st.session_state.hint_move[0], st.session_state.hint_move[1]):
                    button_text = f"⚡{button_text}"
                    
                with cols[x+1]:
                    # 禁用按鈕防禦狀態
                    is_btn_disabled = (st.session_state.phase == "OVER") or \
                                      (st.session_state.phase == "PREPARE" and selected_mode != "🌐 全球線上對戰 (Firebase)" and st.session_state.go_first == "後手 (AI 布置鬼牌)") or \
                                      (selected_mode == "🌐 全球線上對戰 (Firebase)" and not is_my_turn_online)
                                      
                    if st.button(button_text, key=f"tile_{y}_{x}", disabled=is_btn_disabled):
                        st.session_state.prev_board = game.board.copy()
                        
                        if st.session_state.phase == "PREPARE":
                            success, msg = execute_game_step(y, x, side_to_place, is_joker=True)
                            if success:
                                # 💡 修正 1：落子成功後，立刻重新計算棋盤上真實的鬼牌總數，避免 Session State 滯後
                                st.session_state.jokers_placed = np.sum(np.abs(game.board) == 2)
                                
                                if selected_mode == "🌐 全球線上對戰 (Firebase)":
                                    # 💡 修正 2：嚴格遵循規則，先手（玩家1）必須連續放滿 2 張鬼牌
                                    if st.session_state.jokers_placed >= 2:
                                        game.current_player = 2  # 放滿兩張，正式開局並換後手（紅方/玩家2）常規落子
                                        st.session_state.phase = "PLAY"
                                        db.reference(f"rooms/{st.session_state.room_id}").update({
                                            "board": game.board.tolist(),
                                            "status": "PLAY",
                                            "current_player": 2,
                                            "move_log": st.session_state.move_log + ["⚔️ 先手兩張鬼牌已就位！世紀對決正式開打，輪到後手紅方！"]
                                        })
                                    else:
                                        # 未滿兩張，不換手！維持 current_player 為先手，繼續同步雲端
                                        db.reference(f"rooms/{st.session_state.room_id}").update({
                                            "board": game.board.tolist(),
                                            "current_player": game.current_player,
                                            "move_log": st.session_state.move_log
                                        })
                                else:
                                    # 本地模式（單人 VS AI 或 雙人本地）
                                    if st.session_state.jokers_placed >= 2:
                                        game.current_player = 2  # 先手放完兩張鬼牌，換後手
                                        st.session_state.phase = "PLAY"
                                st.rerun()
                            else: 
                                st.toast(f"❌ {msg}")
                        elif st.session_state.phase == "PLAY":
                            success, msg = execute_game_step(y, x, side_to_place)
                            if success:
                                if selected_mode != "🌐 全球線上對戰 (Firebase)" and check_custom_game_over(game.board):
                                    st.session_state.phase = "OVER"
                                st.rerun()
                            else: st.toast(f"❌ {msg}")

    # ---------------------------------------------------------------------
    # 📊 右側欄：分數、狀態與聯網刷新監聽
    # ---------------------------------------------------------------------
    with col_right:
        st.markdown("### 📊 即時戰況")
        st.markdown(f'<div style="background: linear-gradient(135deg, #1D4ED8, #1E3A8A); padding: 10px; border-radius: 8px; color: white; margin-bottom: 8px;"><b>🔵 藍方 / 玩家1</b><br><span style="font-size:22px; font-weight:bold;">{game.scores[1][0]} 條連線</span><br><small>🃏 鬼牌保管: {game.scores[1][1]} 張</small></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background: linear-gradient(135deg, #B91C1C, #7F1D1D); padding: 10px; border-radius: 8px; color: white; margin-bottom: 15px;"><b>🔴 紅方 / 玩家2</b><br><span style="font-size:22px; font-weight:bold;">{game.scores[2][0]} 條連線</span><br><small>🃏 鬼牌保管: {game.scores[2][1]} 張</small></div>', unsafe_allow_html=True)
        
        st.markdown("### 📣 目前狀態")
        if st.session_state.phase == "PREPARE":
            st.warning(f"🎯 階段：請佈置第 {st.session_state.jokers_placed + 1} 張 Joker")
        elif st.session_state.phase == "PLAY":
            if selected_mode == "🌐 全球線上對戰 (Firebase)":
                if is_my_turn_online:
                    st.markdown("<div class='status-text' style='background-color: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid #10B981;'>🟢 輪到你落子了！</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='status-text' style='background-color: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid #F59E0B;'>⏳ 對手思考精算中...</div>", unsafe_allow_html=True)
            else:
                if game.current_player == 1:
                    st.markdown("<div class='status-text' style='background-color: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid #3B82F6;'>🔵 輪到你落子 (藍方)</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='status-text' style='background-color: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid #EF4444;'>🔴 輪到 AI 下棋 ({selected_diff})</div>", unsafe_allow_html=True)
        else:
            # 🏁 賽局結算
            joker_winner = "藍方 / 玩家1" if game.scores[1][0] > game.scores[2][0] else "紅方 / 玩家2"
            winner_code = 1 if game.scores[1][0] > game.scores[2][0] else 2
            if game.scores[1][0] == game.scores[2][0]: 
                joker_winner = "平手"
                winner_code = 0
                
            st.markdown(f"<div class='status-text' style='background-color: #10B981; color: white;'>🏆 最終勝者：{joker_winner}！</div>", unsafe_allow_html=True)
            
            if not st.session_state.leaderboard_updated:
                save_leaderboard(winner_code, is_online=(selected_mode == "🌐 全球線上對戰 (Firebase)"))
                st.session_state.leaderboard_updated = True
                st.session_state.move_log.append(f"🏁 賽局結束！最終勝者為：{joker_winner}")
                st.rerun()

        # 線上模式提供中途退出按鈕
        if selected_mode == "🌐 全球線上對戰 (Firebase)":
            st.write("---")
            if st.button("🚪 退出房間回到大廳", type="secondary", use_container_width=True):
                db.reference(f"rooms/{st.session_state.room_id}").delete()
                st.session_state.net_mode = "LOBBY"
                st.session_state.room_id = None
                st.session_state.my_role = None
                st.rerun()

    # =====================================================================
    # 🤖 AI / 聯網異步監聽後台引擎
    # =====================================================================
    if selected_mode == "🌐 全球線上對戰 (Firebase)" and st.session_state.phase != "OVER":
        # 如果還沒輪到自己，啟動「每秒短輪詢機制」偷看雲端有沒有新棋子
        if not is_my_turn_online:
            time.sleep(1)
            st.rerun()

    elif st.session_state.phase == "PREPARE" and selected_mode != "🌐 全球線上對戰 (Firebase)" and st.session_state.go_first == "後手 (AI 布置鬼牌)":
        with col_right:
            with st.spinner("🤖 AI 正在精算布置鬼牌位置..."):
                time.sleep(0.5)
                st.session_state.prev_board = game.board.copy()
                empty_slots = np.argwhere(game.board == 0)
                idx = np.random.choice(len(empty_slots))
                y, x, side = empty_slots[idx][0], empty_slots[idx][1], np.random.choice([1, -1])
                
                execute_game_step(y, x, side, is_joker=True)
                
                # 💡 修正 3：讓 AI 同樣讀取真實棋盤鬼牌數，確保放滿兩張時精準切換
                st.session_state.jokers_placed = np.sum(np.abs(game.board) == 2)
                if st.session_state.jokers_placed >= 2:
                    st.session_state.game.current_player = 1 # AI（先手）放完兩張，換人類（後手）常規落子
                    st.session_state.phase = "PLAY"
                st.rerun()

    elif st.session_state.phase == "PLAY" and selected_mode != "🌐 全球線上對戰 (Firebase)":
        if selected_mode == "人機對戰 (VS PPO強化學習大腦)" and game.current_player == 2:
            with col_right:
                with st.spinner(f"🤖 {selected_diff} 大腦全速精算中..."):
                    st.session_state.prev_board = game.board.copy()
                    time.sleep(0.3)
                    
                    y, x, side = get_rl_ai_move(game, st.session_state.rl_model)
                    success, msg = execute_game_step(y, x, side)
                    
                    if not success:
                        empty_slots = np.argwhere(game.board == 0)
                        if len(empty_slots) > 0:
                            idx = np.random.choice(len(empty_slots))
                            execute_game_step(empty_slots[idx][0], empty_slots[idx][1], np.random.choice([1, -1]))
                    
                    if check_custom_game_over(game.board): 
                        st.session_state.phase = "OVER"
                    st.rerun()

        elif selected_mode == "神級AI最佳解提示" and st.session_state.hint_move is None:
            with col_right:
                with st.spinner(f"🔮 {selected_diff} 計算最佳解中..."):
                    y, x, side = get_rl_ai_move(game, st.session_state.rl_model)
                    st.session_state.hint_move = (y, x, side)
                    st.rerun()