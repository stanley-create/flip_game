import streamlit as st
import numpy as np
import copy
import time

# 載入核心邏輯
from game import FlipGame, get_god_hint

# =====================================================================
# 🧰 載入外部 CSS 檔案 (寬螢幕佈局)
# =====================================================================
st.set_page_config(page_title="FlipGame 戰術儀表板", layout="wide")

with open("styles.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 初始化 Session State
if 'game' not in st.session_state:
    st.session_state.game = FlipGame()
    st.session_state.phase = "PREPARE"
    st.session_state.jokers_placed = 0
    st.session_state.hint_move = None
    st.session_state.prev_board = np.zeros((4, 4), dtype=int)
    st.session_state.go_first = "先手 (我布置鬼牌)"

game = st.session_state.game

def get_coord_name(y, x):
    return f"{['A', 'B', 'C', 'D'][y]}{x+1}"

# 🆕 新增自訂勝負檢查邏輯：兩張鬼牌沒收（棋盤上完全沒有 2 或 -2）前，都不算結束
def check_custom_game_over(board_matrix):
    # 檢查棋盤內是否還殘留鬼牌 (正面2 或 反面-2)
    has_joker = np.any((board_matrix == 2) | (board_matrix == -2))
    # 如果鬼牌全沒了，或者格子全滿了，遊戲才結束
    is_full = not np.any(board_matrix == 0)
    return (not has_joker) or is_full

# =====================================================================
# ⚙️ 側邊欄
# =====================================================================
with st.sidebar:
    st.title("😈 地獄盲棋控制")
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
    ["人機對戰 (VS 深度5 AI)", "神級AI最佳解提示", "雙人本地對戰"],
    key="play_mode"
)
st.write("---")

# =====================================================================
# 📐 三欄式戰術版面切分
# =====================================================================
col_left, col_center, col_right = st.columns([1, 1.8, 1.2], gap="large")

# ---------------------------------------------------------------------
# ⬅️ 左側欄：動作、落子面向與先後手選擇
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
        if not (selected_mode == "人機對戰 (VS 深度5 AI)" and game.current_player == 2):
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
            
            # 💡 對應 Emoji 屬性標記上色機制
            if val == 0:
                button_text = "⬛"      
            elif val == 1: 
                button_text = "🟦"      
            elif val == -1: 
                button_text = "🟥"      
            elif val == 2:
                button_text = "🃏 J"    
            elif val == -2:
                button_text = "🟥 "     # 鬼牌背面
                
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
                                game.current_player = 2 # 先手玩家擺完，常規換 AI 
                                st.session_state.phase = "PLAY"
                            st.rerun()
                        else: st.toast(f"❌ {msg}")
                    
                    elif st.session_state.phase == "PLAY":
                        if not (selected_mode == "人機對戰 (VS 深度5 AI)" and game.current_player == 2):
                            success, msg = game.step(y, x, side_to_place)
                            if success:
                                st.session_state.hint_move = None
                                # 🆕 核心修正：改用我們自訂的「鬼牌是否被沒收」來判定遊戲結束
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
            st.markdown("<div class='status-text' style='background-color: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid #EF4444;'>🔴 輪到 AI 下棋 (紅方)</div>", unsafe_allow_html=True)
    else:
        # 結算誰勝出
        joker_winner = "藍方" if game.scores[1][0] > game.scores[2][0] else "紅方"
        if game.scores[1][0] == game.scores[2][0]: joker_winner = "平手"
        st.markdown(f"<div class='status-text' style='background-color: #10B981; color: white;'>🏆 鬼牌已全數沒收！最終勝者：{joker_winner}！</div>", unsafe_allow_html=True)

# =====================================================================
# 🧠 AI 背景運算引擎 (包含自訂勝負判定)
# =====================================================================
if st.session_state.phase == "PREPARE" and st.session_state.go_first == "後手 (AI 布置鬼牌)":
    with col_right:
        with st.spinner("🤖 AI 正在精算布置鬼牌位置..."):
            time.sleep(0.5)
            st.session_state.prev_board = game.board.copy()
            move = get_god_hint(game, depth=5)
            if not move:
                empty_slots = np.argwhere(game.board == 0)
                idx = np.random.choice(len(empty_slots))
                move = (empty_slots[idx][0], empty_slots[idx][1], np.random.choice([1, -1]))
            
            game.step(move[0], move[1], move[2], is_joker=True)
            st.session_state.jokers_placed += 1
            
            if st.session_state.jokers_placed >= 2:
                st.session_state.game.current_player = 1 # AI 擺完，強制換玩家開常規第一步
                st.session_state.phase = "PLAY"
            st.rerun()

elif st.session_state.phase == "PLAY":
    if selected_mode == "人機對戰 (VS 深度5 AI)" and game.current_player == 2:
        with col_right:
            with st.spinner("🤖 AI 全速算棋中..."):
                st.session_state.prev_board = game.board.copy()
                time.sleep(0.4)
                move = get_god_hint(game, depth=5)
                if move:
                    game.step(move[0], move[1], move[2])
                    # 🆕 AI 下完後同樣以新規則檢查是否結束
                    if check_custom_game_over(game.board): 
                        st.session_state.phase = "OVER"
                    st.rerun()

    elif selected_mode == "神級AI最佳解提示" and st.session_state.hint_move is None:
        with col_right:
            with st.spinner("🔮 計算最佳解中..."):
                hint = get_god_hint(game, depth=5)
                st.session_state.hint_move = hint
                st.rerun()