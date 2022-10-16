from game import *

AND_STATE = 0 #王手かけてる局面
OR_STATE = 1 # 王手かけられている局面

# アルファベータ法で状態価値計算
def mate_search(state, depth, and_or):
    #print(state)
    # 負けは-1
    if state.is_lose():
        return -1 if and_or == AND_STATE else 1
    # 引き分けも詰まないので-1
    elif state.is_draw():
        return -1 if and_or == AND_STATE else 1
    elif state.is_win():
        return 1 if and_or == AND_STATE else -1
    
    if depth <= 0:
        return -1 

    # 合法手の状態価値の計算
    actions_list = state.check_legal_actions() if and_or == AND_STATE else state.perfect_legal_actions()
    for action in actions_list:
        score = mate_search(state.next(action), depth-1, (and_or ^ 1))
        if and_or == AND_STATE:
            if score == 1:
                return 1
        else:
            if score == -1:
                return -1
    return -1 if and_or == AND_STATE else 1


    # 合法手の状態価値の最大値を返す
    return alpha

# 深さ優先で詰め探索
def mate_action(state):
    best_action = None
    #print(state)
    for action in state.check_legal_actions():
        #print(state.action_str(action))
        score = mate_search(state.next(action), 3, OR_STATE)
        if score == 1:
            best_action = action
            break
    return best_action

def test():
    print("---------------------")
    pieces       = [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 2]
    enemy_pieces = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 2, 0]
    state = State(pieces,enemy_pieces,[])
    print(state)
    print(mate_action(state))

    # 動作確認
if __name__ == '__main__':
    is_end_num = 0
    is_draw_num = 0
    found_mate_num = 0
    sum_depth = 0
    init_key()
    #test()
    #exit(1)
    
    while True:
        # 状態の生成
        state = State()
        ar = state.pieces_array()
        print(f"draw:{is_draw_num} end:{is_end_num} found_mate:{found_mate_num}\r",end="")
        # ゲーム終了までのループ
        while True:
            #print(stateW)
            # ゲーム終了時
            if state.is_lose():
                is_end_num += 1
                print(state)
                print("lose")
                exit(1)
                break
            if state.is_draw():
                is_draw_num += 1
                break
            action = mate_action(state)
            if not action is None:
                found_mate_num += 1
                break
            # 次の状態の取得
            legal_actions = state.perfect_legal_actions()
            if len(legal_actions) == 0:
                is_end_num += 1
                break
            next_action = legal_actions[random.randint(0, len(legal_actions)-1)]
            state = state.next(next_action)

            # 文字列表示
            #print(state)
            #print()
        