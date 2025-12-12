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
    if (m.turn == "B"):
        print(m.turn, m.col, m.row)
        game.play_move(m.turn, m.row, m.col)
    else:
        move_x, move_y = mcts_search(game)
        game.play_move(m.turn, move_x, move_y)
    return {
        "board": game.board,
    }

def mcts_search(root_state, iterations=80):
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
