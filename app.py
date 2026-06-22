import streamlit as st
import numpy as np
import copy
import time
import os
# 💡 導入強化學習載入工具
from stable_baselines3 import PPO

# 載入核心邏輯 (保留 FlipGame 規則結構)
from game import FlipGame

# =====================================================================
# 🧰 載入外部 CSS 檔案 (寬螢幕佈局)
# =====================================================================
st.set_page_config(page_title="FlipGame 戰術儀表板", layout="wide")

# 安全讀取 CSS
if os.path.exists("styles.css"):
    with open("styles.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 💡【難度對應字典】定義不同難度對應的模型檔案名稱
DIFFICULTY_MODELS = {
    "🌱 簡單模式": "best_ppo_blind_v2_model_easy",
    "🔥 中等難度 (94%)": "best_ppo_blind_v2_model_94",
    "💀 地獄模式 (X%)": "best_ppo_blind_v2_model_hell"
}

# 初始化 Session State
if 'game' not in st.session_state:
    st.session_state.game = FlipGame()
    st.session_state.phase = "PREPARE"
    st.session_state.jokers_placed = 0
    st.session_state.hint_move = None
    st.session_state.prev_board = np.zeros((4, 4), dtype=int)
    st.session_state.go_first = "先手 (我布置鬼牌)"
    st.session_state.current_model_path = ""

game = st.session_state.game

def get_coord_name(y, x):
    return f"{['A', 'B', 'C', 'D'][y]}{x+1}"

# 新增自訂勝負檢查邏輯
def check_custom_game_over(board_matrix):
    has_joker = np.any((board_matrix == 2) | (board_matrix == -2))
    is_full = not np.any(board_matrix == 0)
    return (not has_joker) or is_full

# 💡【核心修正】動態載入 PPO 大腦模型函數
def load_selected_model(model_path):
    """ 檢查並動態載入模型，避免重複載入卡頓 """
    if st.session_state.get('current_model_path') == model_path and 'rl_model' in st.session_state:
        return st.session_state.rl_model
        
    if os.path.exists(model_path + ".zip"):
        try:
            model = PPO.load(model_path)
            st.session_state.current_model_path = model_path
            st.session_state.rl_model = model
            return model
        except Exception as e:
            st.sidebar.error(f"❌ 模型損壞或載入出錯: {str(e)}")
            return None
    else:
        st.session_state.rl_model = None
        st.session_state.current_model_path = ""
        return None

# 💡 強化學習大腦決策引擎
def get_rl_ai_move(game_instance, model):
    """ 讓指定難度的大腦戴上盲棋面罩看盤面，並吐出最佳落子動作 """
    if model is None:
        # 如果模型沒載入成功，退化成隨機落子安全防線
        empty_slots = np.argwhere(game_instance.board == 0)
        idx = np.random.choice(len(empty_slots))
        return empty_slots[idx][0], empty_slots[idx][1], np.random.choice([1, -1])
    
    # 🎭 1. 幫大腦戴上盲棋面罩 (將真實棋盤的 -2 鬼牌背面轉為 -1)
    masked_obs = game_instance.board.copy()
    masked_obs[masked_obs == -2] = -1
    masked_obs = masked_obs.astype(np.float32)
    
    # 2. 讓 PPO 大腦進行預測 (predict)
    action, _ = model.predict(masked_obs, deterministic=True)
    
    # 3. 將大腦輸出的 0~31 動作編號，還原回 (y, x, side) 座標規格
    action = int(action)
    y = action // 8
    x = (action % 8) // 2
    side = 1 if action % 2 == 0 else -1
    return y, x, side

# =====================================================================
# ⚙️ 側邊欄 (新增難度切換)
# =====================================================================
with st.sidebar:
    st.title("😈 地獄盲棋控制")
    st.write("---")
    
    # 🆕 難度選擇下拉選單
    selected_diff = st.selectbox(
        "🧠 選擇 AI 大腦難度等級",
        list(DIFFICULTY_MODELS.keys()),
        index=1 # 預設選擇 94% 中等難度
    )
    
    # 動態載入當前選擇的難度大腦
    target_path = DIFFICULTY_MODELS[selected_diff]
    current_brain = load_selected_model(target_path)
    
    if current_brain is not None:
        st.success(f"🟢 {selected_diff} 大腦已成功連線！")
    else:
        st.error(f"⚠️ 找不到 `{target_path}.zip`！\n該難度下 AI 將流於隨機落子。")
        
    st.write("---")
    st.info("💡 盲棋規則：\n當鬼牌翻轉成【反面】時，會變成灰紅色且褪去文字，完美偽裝成一般背面方塊。")
    if st.button("🔄 重置整個賽局", type="primary", use_container_width=True):
        st.session_state.game = FlipGame()
        st.session_state.phase = "PREPARE"
        st.session_state.jokers_placed = 0
        st.session_state.hint_move = None
        st.session_state.prev_board = np.zeros((4, 4), dtype=int)
        st.rerun()

# 頂部模式切換
st.title("🧱 賽博正反棋 : TILE MATRIX")
selected_mode = st.selectbox(
    "🤖 核心對弈模式切換", 
    ["人機對戰 (VS PPO強化學習大腦)", "神級AI最佳解提示", "雙人本地對戰"],
    key="play_mode"
)
st.write("---")

# =====================================================================
# 📐 三欄式戰術版面切分
# =====================================================================
col_left, col_center, col_right = st.columns([1, 1.8, 1.2], gap="large")

# ---------------------------------------------------------------------
# ⬅️ 左側欄
# ---------------------------------------------------------------------
with col_left:
    st.markdown("### 🎛️ 戰術動作")
    
    if st.session_state.phase == "PREPARE":
        st.write("⏱️ **對局順序設定**")
        prev_go_first = st.session_state.go_first
        st.session_state.go_first = st.radio(
            "請選擇你的順位：",
            ["先手 (我布置鬼牌)", "後手 (AI 布置鬼牌)"],
            label_visibility="collapsed"
        )
        if prev_go_first != st.session_state.go_first and st.session_state.jokers_placed == 0:
            st.rerun()
            
        st.write("---")
        
        if st.session_state.go_first == "先手 (我布置鬼牌)":
            st.write("🔧 **設定鬼牌初始面向**")
            side_to_place = st.radio(
                "選擇面向：", [1, -1], 
                format_func=lambda x: "🟦 藍色 (正面方塊)" if x==1 else "🟥 灰紅色 (隱藏方塊)"
            )
        else:
            st.info("🤖 AI 正在全權負責布置鬼牌...")
            side_to_place = -1 
            
    elif st.session_state.phase == "PLAY":
        st.markdown(f"⏱️ 順位：`{st.session_state.go_first}`")
        st.write("---")
        
        st.write("🎨 **選擇本次落子面向**")
        if not (selected_mode == "人機對戰 (VS PPO強化學習大腦)" and game.current_player == 2):
            side_to_place = st.radio(
                "選擇面向：", [1, -1], 
                format_func=lambda x: "🟦 淺藍色 (正面)" if x==1 else "🟥 灰紅色 (反面)"
            )
        else:
            st.info("🤖 AI 正在精算中...")
            side_to_place = 1
    else:
        st.success("🏁 賽局已結束")
        side_to_place = 1

# ---------------------------------------------------------------------
# 🎯 中央欄：核心方塊棋盤 
# ---------------------------------------------------------------------
with col_center:
    cols_top = st.columns([0.5, 1, 1, 1, 1])
    for x in range(4):
        cols_top[x+1].markdown(f"<div class='axis-label'>{x+1}</div>", unsafe_allow_html=True)

    for y in range(4):
        cols = st.columns([0.5, 1, 1, 1, 1])
        cols[0].markdown(f"<div class='axis-label'>{['A', 'B', 'C', 'D'][y]}</div>", unsafe_allow_html=True)
        
        for x in range(4):
            val = game.board[y, x]
            
            if val == 0: button_text = "⬛"      
            elif val == 1: button_text = "🟦"      
            elif val == -1: button_text = "🟥"      
            elif val == 2: button_text = "🃏 J"    
            elif val == -2: button_text = "🟥 "     # 鬼牌背面
                
            if selected_mode == "神級AI最佳解提示" and st.session_state.hint_move and (y, x) == (st.session_state.hint_move[0], st.session_state.hint_move[1]):
                button_text = f"⚡{button_text}"
                
            with cols[x+1]:
                is_btn_disabled = (st.session_state.phase == "OVER") or \
                                  (st.session_state.phase == "PREPARE" and st.session_state.go_first == "後手 (AI 布置鬼牌)")
                                  
                if st.button(button_text, key=f"tile_{y}_{x}", disabled=is_btn_disabled):
                    st.session_state.prev_board = game.board.copy()
                    
                    if st.session_state.phase == "PREPARE":
                        success, msg = game.step(y, x, side_to_place, is_joker=True)
                        if success:
                            st.session_state.jokers_placed += 1
                            if st.session_state.jokers_placed >= 2:
                                game.current_player = 2 
                                st.session_state.phase = "PLAY"
                            st.rerun()
                        else: st.toast(f"❌ {msg}")
                    
                    elif st.session_state.phase == "PLAY":
                        if not (selected_mode == "人機對戰 (VS PPO強化學習大腦)" and game.current_player == 2):
                            success, msg = game.step(y, x, side_to_place)
                            if success:
                                st.session_state.hint_move = None
                                if check_custom_game_over(game.board): 
                                    st.session_state.phase = "OVER"
                                st.rerun()
                            else: st.toast(f"❌ {msg}")

# ---------------------------------------------------------------------
# 📊 右側欄：分數與狀態看板
# ---------------------------------------------------------------------
with col_right:
    st.markdown("### 📊 即時戰況")
    st.markdown(f'<div style="background: linear-gradient(135deg, #1D4ED8, #1E3A8A); padding: 10px; border-radius: 8px; color: white; margin-bottom: 8px;"><b>🔵 藍方 (正面)</b><br><span style="font-size:22px; font-weight:bold;">{game.scores[1][0]} 條連線</span><br><small>🃏 鬼牌保管: {game.scores[1][1]} 張</small></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background: linear-gradient(135deg, #B91C1C, #7F1D1D); padding: 10px; border-radius: 8px; color: white; margin-bottom: 15px;"><b>🔴 紅方 (反面)</b><br><span style="font-size:22px; font-weight:bold;">{game.scores[2][0]} 條連線</span><br><small>🃏 鬼牌保管: {game.scores[2][1]} 張</small></div>', unsafe_allow_html=True)
    
    st.markdown("### 📣 目前狀態")
    if st.session_state.phase == "PREPARE":
        if st.session_state.go_first == "先手 (我布置鬼牌)":
            st.warning(f"🎯 請玩家佈置第 {st.session_state.jokers_placed + 1} 張 Joker")
        else:
            st.error(f"🤖 AI 正在佈置第 {st.session_state.jokers_placed + 1} 張 Joker...")
    elif st.session_state.phase == "PLAY":
        if game.current_player == 1:
            st.markdown("<div class='status-text' style='background-color: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid #3B82F6;'>🔵 輪到你落子 (藍方)</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='status-text' style='background-color: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid #EF4444;'>🔴 輪到 AI 下棋 ({selected_diff})</div>", unsafe_allow_html=True)
    else:
        joker_winner = "藍方" if game.scores[1][0] > game.scores[2][0] else "紅方"
        if game.scores[1][0] == game.scores[2][0]: joker_winner = "平手"
        st.markdown(f"<div class='status-text' style='background-color: #10B981; color: white;'>🏆 鬼牌已全數沒收！最終勝者：{joker_winner}！</div>", unsafe_allow_html=True)

# =====================================================================
# 🧠 AI 背景運算引擎 (💡 支援動態切換大腦難度)
# =====================================================================
if st.session_state.phase == "PREPARE" and st.session_state.go_first == "後手 (AI 布置鬼牌)":
    with col_right:
        with st.spinner("🤖 AI 正在精算布置鬼牌位置..."):
            time.sleep(0.5)
            st.session_state.prev_board = game.board.copy()
            
            # 布置鬼牌階段退化隨機安全機制
            empty_slots = np.argwhere(game.board == 0)
            idx = np.random.choice(len(empty_slots))
            move = (empty_slots[idx][0], empty_slots[idx][1], np.random.choice([1, -1]))
            
            game.step(move[0], move[1], move[2], is_joker=True)
            st.session_state.jokers_placed += 1
            
            if st.session_state.jokers_placed >= 2:
                st.session_state.game.current_player = 1 
                st.session_state.phase = "PLAY"
            st.rerun()

elif st.session_state.phase == "PLAY":
    # 💡 核心對決區：當輪到 AI (Player 2)
    if selected_mode == "人機對戰 (VS PPO強化學習大腦)" and game.current_player == 2:
        with col_right:
            with st.spinner(f"🤖 {selected_diff} 大腦全速精算中..."):
                st.session_state.prev_board = game.board.copy()
                time.sleep(0.3)
                
                # 💡【核心改動】使用當前在側邊欄選擇的強化學習大腦模型做出決策
                y, x, side = get_rl_ai_move(game, st.session_state.rl_model)
                
                success, msg = game.step(y, x, side)
                
                # 安全防禦機制：防止 AI 預測出不合法的步數導致畫面死鎖
                if not success:
                    empty_slots = np.argwhere(game.board == 0)
                    if len(empty_slots) > 0:
                        idx = np.random.choice(len(empty_slots))
                        game.step(empty_slots[idx][0], empty_slots[idx][1], np.random.choice([1, -1]))
                
                if check_custom_game_over(game.board): 
                    st.session_state.phase = "OVER"
                st.rerun()

    # 💡 提示模式區
    elif selected_mode == "神級AI最佳解提示" and st.session_state.hint_move is None:
        with col_right:
            with st.spinner(f"🔮 {selected_diff} 計算最佳解中..."):
                y, x, side = get_rl_ai_move(game, st.session_state.rl_model)
                st.session_state.hint_move = (y, x, side)
                st.rerun()