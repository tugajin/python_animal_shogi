# ====================
# セルフプレイ部
# ====================

# パッケージのインポート
from game import State
from pv_mcts import pv_mcts_scores
from dual_network import DN_OUTPUT_SIZE
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
import os
import torch
from dual_network import *
import concurrent.futures
import copy
from multiprocessing import Process

# パラメータの準備
PROCESS_NUM = 4
SP_GAME_COUNT = int(500 / PROCESS_NUM) # セルフプレイを行うゲーム数（本家は25000）
SP_TEMPERATURE = 1.0 # ボルツマン分布の温度パラメータ

# 先手プレイヤーの価値
def first_player_value(ended_state):
    # 1:先手勝利, -1:先手敗北, 0:引き分け
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

# 学習データの保存
def write_data(history,id):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True) # フォルダがない時は生成
    if id == "":
        path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
            now.year, now.month, now.day, now.hour, now.minute, now.second)
    else:
        path = './data/{}.tmp'.format(id)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)

# 1ゲームの実行
def play(model,device):
    # 学習データ
    history = []

    # 状態の生成
    state = State()

    while True:
        # ゲーム終了時
        if state.is_done():
            break
        #print("do")        
        # 合法手の確率分布の取得
        scores = pv_mcts_scores(model,device, state, SP_TEMPERATURE)

        # 学習データに状態と方策を追加
        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([state.pieces_array(), policies, None])

        # 行動の取得
        action = np.random.choice(state.legal_actions(), p=scores)
    
        # 次の状態の取得
        state = state.next(action)
        
    # 学習データに価値を追加
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value
    return history

# セルフプレイ
def self_play(id):
    # 学習データ
    history = []
    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # なぜかcpuのほうが早い。
    #device = 'cpu'
    
    model = DualNet()
    model.load_state_dict(torch.load('./model/best.h5'))
    model = model.double()
    model = model.to(device)
    model.eval()
    
    # 複数回のゲームの実行
    for i in range(SP_GAME_COUNT):
        # 1ゲームの実行
        h = play(model,device)
        
        history.extend(h)

        # 出力
        print('\r{}SelfPlay {}/{}'.format(id, i+1, SP_GAME_COUNT), end='')
    print('')
    
    write_data(history,id)
    #return history

# 並列化セルフプレイ
def paralell_self_play():
    history = []
    process_list = []
    for i in range(PROCESS_NUM):
        p = Process(target=self_play, args=(i,))
        process_list.append(p)
        
    for i in range(PROCESS_NUM):
        print(str(i) + " start")
        process_list[i].start()
    for i in range(PROCESS_NUM):
        process_list[i].join()
    
    # historyデータをマージ
    xs_all = []
    y_policies_all = []
    y_values_all = []
    for i in range(PROCESS_NUM):
        history_path = sorted(Path('./data/').glob(str(i) + '.tmp'))[-1]
        with history_path.open(mode='rb') as f:
            data = pickle.load(f)
        xs, y_policies, y_values = zip(*data)
        xs_all.extend(xs)
        y_policies_all.extend(y_policies)
        y_values_all.extend(y_values)
        os.remove('./data/' + str(i) + ".tmp")
    history = [(xs_all[i],y_policies_all[i],y_values_all[i]) for i in range(len(xs_all))]
    
    write_data(history,"")
    
    
# 動作確認
if __name__ == '__main__':
    paralell_self_play()
    #self_play()
