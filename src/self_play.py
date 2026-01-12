import os
import math
from go import Go
from mcts_node import MCTSNode

def mcts_search(root_state, game_num, turn, iterations=400):
    root = MCTSNode(root_state)

    for i in range(iterations): # how much to look ahead by?
        node = root

        while(node.children != []):
            node = node.best_child()

        # Expansion
        if (node.children == []):
            p, v = node.expand()
        

        # Backpropagation
        node.backpropagate(int(v))
    
    best_child = root.best_child()
    if(turn <= 30): tau = 1
    else: tau = 0.5
    pi = [0] * 81
    for i in range(9):
        for j in range(9):
            for child in root.children:
                if (child.action == (i,j)):
                    pi[i*9+j] = (child.visits ** (1/tau) / root.visits ** (1/tau))
                    break
    with open(f"games/game{game_num}/turn{best_child.turn_number}.txt", "a") as f:
        f.write(best_child.game_state.print_board())
        f.write(str(pi))
        f.write("\n")

    # print(len(root.children))
    # for child in root.children:
    #     print("-------")
    #     print(child.action)
    #     print(child.prior_prob)
    #     print (child.action_val + 1.4*child.prior_prob*math.sqrt(root.visits)/(1 + child.visits))
    # print (best_child.action, best_child.action_val + 1.4*best_child.prior_prob*math.sqrt(root.visits)/(1 + best_child.visits))

    return best_child.action  # Return best move

def self_play(game_num):
    print("running")
    os.makedirs(f"games/game{game_num}")
    game = Go(9,9)
    for i in range(128):
        move_x, move_y = mcts_search(game, game_num, i)
        game.play_move(game.curr_player, move_x, move_y)
        game.turn_number += 1
    white, black = game.score()
    winner = 0 # 1 for black, -1 for white
    if(white < black):
        winner = 1
    elif(white > black):
        winner = -1

    for i in range(1, 129):
        with open(f"games/game{game_num}/turn{i}.txt", "a") as f:
            if(i % 2 == 1):
                f.write(str(winner))
            else:
                f.write(str(winner * -1))
