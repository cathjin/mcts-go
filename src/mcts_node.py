import random
import copy
import torch
import numpy as np 
import math

from go import Go
from neural_network import NeuralNetwork

NUM_COLS = 9
NUM_ROWS = 9

class MCTSNode:
    def __init__(self, game_state : Go, action = None, parent = None):
        self.game_state = game_state
        self.action = action
        self.untried_actions = self.get_actions()

        self.parent = parent
        self.children = []

        self.turn_number = game_state.turn_number + 1

        self.visits = 0
        self.action_val = 0
        self.prior_prob = 0
        self.value = 0

    # remove possibility for returning moves that are captured + ko
    def get_actions(self):
        actions = []
        for i in range(NUM_ROWS):
            for j in range(NUM_COLS):
                player = self.game_state.curr_player
                op = "B" if player == "W" else "W"
                if(self.game_state.board[i][j] == "O" and self.game_state.check_captured(player, op, i, j, []) is False):
                    actions.append((i,j))
        return actions
    
    def best_child(self, c=1.4):
        return max(self.children, key=lambda child:
                child.action_val + c*child.prior_prob*math.sqrt(self.visits)/(1 + child.visits))

    def expand(self):
        p, v = self.evaluate()
        for i in range(9):
            for j in range(9):
                if((i,j) not in self.untried_actions): 
                    continue
                new_state = copy.deepcopy(self.game_state)
                # if(self.game_state.curr_player == "B"):
                #     new_state.curr_player = "W"
                # else:
                #     new_state.curr_player = "B"
                new_state.play_move(new_state.curr_player, i,j)
                # new_state.board[i][j] = new_state.curr_player
                child = MCTSNode(new_state, parent=self, action=(i, j))
                child.prior_prob = p[i * 9 + j]
                self.children.append(child)
        self.value = v
        return p, v

    
    def evaluate(self): # rename
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        model = NeuralNetwork()
        model.load_state_dict(torch.load("model_params.pth", weights_only=True))
        model.eval()
        model = model.to(device)
        board = self.game_state.board
        int_board = copy.deepcopy(board)
        for i in range(9):
            for j in range(9):
                if board[i][j] == "O":
                    int_board[i][j] = 0
                elif board[i][j] == "B":
                    int_board[i][j] = 1
                else:
                    int_board[i][j] = 2
        state = torch.tensor([int_board], dtype = torch.float32).unsqueeze(0)
        state = state.to(device)
        with torch.no_grad():
            p, v = model(state)
        p = p.detach().cpu().numpy()[0]
        return p, v

    def backpropagate(self, value):
        self.action_val = ((self.action_val * self.visits) + value)/(self.visits + 1)
        self.visits += 1

        if self.parent:
            self.parent.backpropagate(value)
    
 # resign condition