from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.go import Go
from src.mcts_node import MCTSNode
import uvicorn
import random

random.seed(42)

# Create ONE game that persists between requests
game = Go(9,9)
root = MCTSNode(game)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Change for production as needed.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

class Move(BaseModel):
    turn : str
    col: int
    row: int


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/move")
async def play_move(m: Move):
    global game
    if (m.turn == "B"):
        print(m.turn, m.col, m.row)
        game.play_move(m.turn, m.row, m.col)
    else:
        best_child = mcts_search(game)
        root = best_child
        root.parent = None
        game = root.game_state
        # game.play_move(m.turn, move_x, move_y)
    return {
        "board": game.board,
    }

def mcts_search(root_state, iterations=200):
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
    # for child in root.children:
    #     print (child.action_val + 1.4*child.prior_prob/(1 + child.visits))
    # print (root.best_child().action, child.action_val + 1.4*child.prior_prob/(1 + child.visits))

    return best_child  # Return best move
