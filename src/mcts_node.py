import random
import copy
import math
from go import Go

NUM_COLS = 9
NUM_ROWS = 9

class MCTSNode:
    def __init__(self, game_state : Go, action = None, parent = None):
        self.game_state = game_state
        self.action = action

        self.parent = parent
        self.children = []

        self.visits = 0
        self.score = 0
        self.untried_actions = self.get_actions()
    
    def get_actions(self):
        actions = []
        for i in range(NUM_ROWS):
            for j in range(NUM_COLS):
                if(self.game_state.board[i][j] == "O"):
                    # self.game_state.board[i][j] = "W"
                    # if(not self.game_state.check_captured("W", "B", i, j, [])):
                    actions.append((i,j))
                    # self.game_state.board[i][j] = "O"

        return actions
        # doesn't consider ko
    
    def best_child(self, c=1.4):
        return max(self.children, key=lambda child:
                (child.score / child.visits) +
                c * math.sqrt(math.log(self.visits) / child.visits))

    def expand(self):
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        new_state = copy.deepcopy(self.game_state)
        new_state.board[action[0]][action[1]] = new_state.curr_player
        child = MCTSNode(new_state, parent=self, action=action)
        self.children.append(child)
        return child

    
    def rollout(self):
        state = copy.deepcopy(self.game_state)
        for i in range(10000): # how many...
            actions = self.get_actions()
            if actions:
                move_x, move_y = random.choice(actions)
                state.board[move_x][move_y] = state.curr_player
                state.curr_player = "B" if state.curr_player == "W" else "W"
        white_score, black_score = state.score()
         # want to compare to black?
        # return white_score/(NUM_COLS*NUM_ROWS)
        return white_score
        # return (white_score - black_score)
        # return (white_score - black_score)/(NUM_COLS*NUM_ROWS)
        
    def backpropagate(self, score):
        self.visits += 1
        self.score += score
        if self.parent:
            self.parent.backpropagate(score)
    
    