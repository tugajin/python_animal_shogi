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
    epoch_num = 5
    batch_size = 512
    if len(args) >= 4:
        self_play_num = int(args[1])
        epoch_num = int(args[2])
        batch_size = int(args[3])

    # デュアルネットワークの作成
    print(f"selfplay:{self_play_num}")
    print(f"epoch:{epoch_num}")
    print(f"batch:{batch_size}")
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
