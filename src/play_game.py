from go import Go
from mcts_node import MCTSNode
    
def get_input():
    valid = False
    move_x = input("Black move x: ")
    move_y = input("Black move y: ")
    move_x = int(move_x)
    if move_y == "J": 
        move_y = "I"
    if move_x < 1 or move_x > 9 or move_y > 'J' or move_y < 'A':
        valid = False
        print("INVALID INPUT TRY AGAIN")
    else:
        valid = True
    return valid, move_x, move_y

def play_game():
    game = Go(9,9)
    while(True):
        if(game.curr_player == "B"):
            valid = False
            while(not valid):
                valid, move_x, move_y = get_input()
            print(9 - int(move_x), move_y)
            
        else:
            move_x, move_y = mcts_search(game)
            mcts_move_y = chr(ord('A') + move_y)
            if mcts_move_y == "I":
                mcts_move_y = "J"
            print(9 - int(move_x), mcts_move_y)
        game.play_move(game.curr_player, move_x, move_y)
        game.curr_player = "W" if game.curr_player == "B" else "B"

def mcts_search(root_state, iterations=8000):
    root = MCTSNode(root_state)

    for i in range(iterations): # how much to look ahead by?
        node = root

        while(node.children != [] and node.untried_actions == []):
            node = node.best_child()

        # Expansion
        if (node.untried_actions):
            node = node.expand()
        
        # Simulation
        score = node.rollout() # remove rollout??

        # Backpropagation
        node.backpropagate(score)
    print(len(root.children))
    for child in root.children:
        print (child.action, child.score/child.visits)
    print (root.best_child(c=0).action, root.best_child(c=0).score/root.best_child(c=0).visits)
    return root.best_child(c=0).action  # Return best move

if __name__ == "__main__":
    play_game()

    # putting pieces down so that you get captured???