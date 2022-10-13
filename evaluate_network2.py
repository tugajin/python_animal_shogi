# ====================
# 新パラメータ評価部
# ====================

# パッケージのインポート
from game import State
from pv_mcts import pv_mcts_action
from pv_ubfm import *
from pathlib import Path
from shutil import copy
import numpy as np
from single_network import *
import shutil

# パラメータの準備
EN_GAME_COUNT = 10 # 1評価あたりのゲーム数（本家は400）
EN_TEMPERATURE = 0.7 # ボルツマン分布の温度

# 先手プレイヤーのポイント
def first_player_point(ended_state):
    # 1:先手勝利, 0:先手敗北, 0.5:引き分け
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

# 1ゲームの実行
def play(next_actions,device):
    # 状態の生成
    state = State()

    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 行動の取得
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state,device)

        # 次の状態の取得
        state = state.next(action)

    # 先手プレイヤーのポイントを返す
    return first_player_point(state)

# ベストプレイヤーの交代
def update_best_player():
    shutil.copy('./model/latest_single.h5', './model/best_single.h5')
    print('Change BestPlayer')

# ネットワークの評価
def evaluate_network():
    # 最新プレイヤーのモデルの読み込み
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    model0 = DualNet()
    model0.load_state_dict(torch.load('./model/latest.h5',device))
    model0 = model0.double()
    model0 = model0.to(device)
    model0.eval()
    

    # ベストプレイヤーのモデルの読み込み
    model1 = DualNet()
    model1.load_state_dict(torch.load('./model/best.h5',device))
    model1 = model1.double()
    model1 = model1.to(device)
    model1.eval()

    
    # PV MCTSで行動選択を行う関数の生成
    next_action0 = pv_mcts_action(model0,device, EN_TEMPERATURE)
    next_action1 = pv_mcts_action(model1,device, EN_TEMPERATURE)
    next_actions = (next_action0, next_action1)

    # 複数回の対戦を繰り返す
    total_point = 0
    for i in range(EN_GAME_COUNT):
        # 1ゲームの実行
        if i % 2 == 0:
            total_point += play(next_actions,device)
        else:
            total_point += 1 - play(list(reversed(next_actions)),device)

        # 出力
        print('\rEvaluate {}/{}'.format(i + 1, EN_GAME_COUNT), end='')
    print('')

    # 平均ポイントの計算
    average_point = total_point / EN_GAME_COUNT
    print('AveragePoint', average_point)


    # ベストプレイヤーの交代
    if average_point > 0.5:
        update_best_player()
        return True
    else:
        return False

def evaluate_problem():
    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5',device))
    model = model.to(device)
    model.eval()

    # 状態の生成
    state = State()
    print(state)
    score, values = pv_ubfm_scores(model, state, device, EN_TEMPERATURE) 
    moves = state.legal_actions()
    for i in range(len(moves)):
        print(state.action_str(moves[i]) + ":" + str(score[i]))
    print(values)
    print("---------------------")
    pieces       = [0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 4, 3, 0, 0, 0]
    enemy_pieces = [0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 1, 3, 0, 0, 0]
    state = State(pieces,enemy_pieces,[])
    print(state)
    score, values = pv_ubfm_scores(model, state, device, EN_TEMPERATURE) 
    moves = state.legal_actions()
    for i in range(len(moves)):
        print(state.action_str(moves[i]) + ":" + str(score[i]))
    print(values)
    print("---------------------")
    pieces       = [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 2]
    enemy_pieces = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 2, 0]
    state = State(pieces,enemy_pieces,[])
    print(state)
    score, values = pv_ubfm_scores(model, state, device, EN_TEMPERATURE) 
    moves = state.legal_actions()
    for i in range(len(moves)):
        print(state.action_str(moves[i]) + ":" + str(score[i]))
    print(values)
    print("---------------------")
    pieces       = [0, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1]
    enemy_pieces = [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0]
    state = State(pieces,enemy_pieces,[])
    print(state)
    score, values = pv_ubfm_scores(model, state, device, EN_TEMPERATURE) 
    moves = state.legal_actions()
    for i in range(len(moves)):
        print(state.action_str(moves[i]) + ":" + str(score[i]))
    print(values)
    print("---------------------")
                   #0  1  2  3  4  5  6  7  8  9  10 11 ひ ぞ き
    pieces       = [0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 3, 0, 0, 0, 0]
                  #11 10  9  8  7  6  5  4  3  2  1  0 ひ ぞ き
    enemy_pieces = [0, 0, 0, 0, 1, 0, 0, 0, 3, 2, 4, 0, 1, 0, 0]
    state = State(pieces,enemy_pieces,[])
    print(state)
    score, values = pv_ubfm_scores(model, state, device, EN_TEMPERATURE) 
    moves = state.legal_actions()
    for i in range(len(moves)):
        print(state.action_str(moves[i]) + ":" + str(score[i]))
    print(values)
    print("---------------------")
# 動作確認
if __name__ == '__main__':
    init_key()
    evaluate_problem()
    #evaluate_network()
