import os
import math
from go import Go
from mcts_node import MCTSNode
import time


def mcts_search(root, game_num, turn, model, iterations=400):
    for i in range(iterations):  # how much to look ahead by?
        node = root

        while node.children != []:
            node = node.best_child()

        # Expansion
        # start_time = time.perf_counter()
        if node.children == []:
            p, v = node.expand(model)

        # end_time = time.perf_counter()
        # elapsed_time = end_time - start_time

        # print(f"Elapsed time: {elapsed_time:.4f} seconds")
        # Backpropagation
        node.backpropagate(int(v))
    if(root.children):
        best_child = root.best_child()
        if turn <= 30:
            tau = 1
        else:
            tau = 0.5
        pi = [0] * 81
        for child in root.children:
            index = child.action[0] * 9 + child.action[1]
            pi[index] = child.visits ** (1 / tau) / root.visits ** (1 / tau)
        with open(f"games/game{game_num}/turn{turn + 1}.txt", "a") as f:
            f.write(best_child.game_state.print_board())
            f.write(str(pi))
            f.write("\n")
        return best_child  # Return best move
    else:
        return False


    # print(len(root.children))
    # for child in root.children:
    #     print("-------")
    #     print(child.action)
    #     print(child.prior_prob)
    #     print (child.action_val + 1.4*child.prior_prob*math.sqrt(root.visits)/(1 + child.visits))
    # print (best_child.action, best_child.action_val + 1.4*best_child.prior_prob*math.sqrt(root.visits)/(1 + best_child.visits))

    


def self_play(game_num, model):
    print("running")
    os.makedirs(f"games/game{game_num}")
    game = Go(9)
    root = MCTSNode(game)
    num_moves = 0
    for i in range(128):
        num_moves = i
        best_child = mcts_search(root, game_num, i, model)
        if(best_child):
            root = best_child
            root.parent = None
            # move_x, move_y = best_child.action
            # game.play_move(game.curr_player, move_x, move_y)
            game = root.game_state
            game.turn_number += 1
        else:
            break
    white, black = game.score()
    winner = 0  # 1 for black, -1 for white
    if white < black:
        winner = 1
    elif white > black:
        winner = -1

    for i in range(1, num_moves + 1):
        with open(f"games/game{game_num}/turn{i}.txt", "a") as f:
            if i % 2 == 1:
                f.write(str(winner))
            else:
                f.write(str(winner * -1))
    return num_moves
