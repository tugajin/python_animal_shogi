# ====================
# 学習サイクルの実行
# ====================

# パッケージのインポート
from single_network import single_network
from pv_descent import *
from train_network4 import train_network
from evaluate_network2 import *
from evaluate_best_player import *
import multiprocessing as mp
import sys


if __name__ == '__main__':

    mp.set_start_method('spawn')
    init_key()

    args = sys.argv
    self_play_num = 10
    epoch_num = 15
    batch_size = 128
    if len(args) >= 4:
        self_play_num = args[1]
        epoch_num = args[2]
        batch_size = args[3]

    # デュアルネットワークの作成
    single_network()
    
    for i in range(25):
        print('Train',i,'====================')
        # セルフプレイ部
        self_play(self_play_num)

        # パラメータ更新部
        train_network(epoch_num,batch_size)

        # 新パラメータ評価部
        #evaluate_network()
        evaluate_problem()
        evaluate_best_player()
        update_best_player()