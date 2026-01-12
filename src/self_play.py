import os

from go import Go
from mcts_node import MCTSNode

def mcts_search(root_state, game_num, iterations=100):
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
    
    best_child = root.best_child(c=0)
    with open(f"games/game{game_num}/turn{best_child.turn_number}.txt", "a") as f:
        f.write(best_child.game_state.print_board())
        f.write(str(p))
        f.write("\n")

    # print(len(root.children))
    # for child in root.children:
    #     print("-------")
    #     print(child.action)
    #     print(child.prior_prob)
    #     print (child.action_val + 1.4*child.prior_prob/(1 + child.visits))
    # print (root.best_child(c=0).action, child.action_val + 1.4*child.prior_prob/(1 + child.visits))

    return best_child.action  # Return best move

def self_play(game_num):
    print("running")
    os.makedirs(f"games/game{game_num}")
    game = Go(9,9)
    for i in range(128):
        move_x, move_y = mcts_search(game, game_num)
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
