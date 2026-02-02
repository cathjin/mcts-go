import os
from go import Go
from mcts_node import MCTSNode
import torch
from typing import Union

def mcts_search(root : MCTSNode, game_num : int, 
                turn : int, model : torch.nn.Module, 
                iterations : int = 400) -> Union[MCTSNode, bool]:
    for i in range(iterations):
        node = root
        
        # find best leaf node to expand
        while node.children != []:
            node = node.best_child()

        # expand leaf node
        if node.children == []:
            p, v = node.expand(model)

        # propagate results back up the tree
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

def self_play(game_num : int, model : torch.nn.Module) -> int:
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
            game = root.game_state
            game.turn_number += 1
        else:
            break
    white, black = game.score()
    winner = 0  # 1 for black, -1 for white if black wins
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
