import os
import torch
import numpy as np
from numpy.linalg import norm
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F

from neural_network import NeuralNetwork
from self_play import self_play

class SelfPlayDataset(Dataset):
    def __init__(self, turn_files, game_dir):
        self.turn_files = turn_files
        self.game_dir = game_dir
    
    def __len__(self):
        return len(self.turn_files)
    
    def __getitem__(self, index):
        game_path = os.path.join(self.game_dir, self.turn_files[index])
        state_string = ""
        move_prob = ""
        with open(game_path, 'r') as file:
            for i in range(17):
                state_string+=(next(file))
            for i in range(14):
                move_prob += next(file)
            
            win = float(next(file))
        
        lines = state_string.strip().split("\n")
        state = []
        for line in lines:
            if "|" not in line:  # This is a data line
                row = line.split(" - ")  # Split by " - " to get individual elements
                for i in range(len(row)):
                    if(row[i] == "O"): row[i] = 0
                    elif(row[i] == "B"): row[i] = 1
                    else: row[i] = 2
                state.append(row)
        move_prob = move_prob.strip().strip("[]").split()
        for i in range(len(move_prob)):
            move_prob[i] = float(move_prob[i])
        state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0)
        move_prob = torch.tensor(move_prob, dtype=torch.float32)
        win = torch.tensor(win, dtype=torch.float32)
        expected = (move_prob, win)
        return state, expected

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")
model = NeuralNetwork()#.to(device)
model.load_state_dict(torch.load("model_params.pth", weights_only=True))
print(model)

learning_rate = 1e-2
batch_size = 32
epochs = 5

def loss_fnc(pred, y, params):
    output_p, output_v = pred
    target_p, target_v = y
    loss = F.mse_loss(output_v, target_v) #- F.cross_entropy(output_p, target_p) + 1.4 * (sum(p.pow(2).sum() for p in params) ** 2)
    print("LOSS", loss)
    return loss

loss_fn = nn.MSELoss()

    
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X,y) in enumerate(dataloader):
        p, z = model(X)
        pi, v = y
        print(z,v)
        loss = loss_fn(z, v)
        print("LOSS", loss)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(name, param.grad.abs().mean())
            else:
                print(name, "has no gradient")
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

torch.autograd.set_detect_anomaly(True)
def train():
    for game_num in range(30):
        self_play(game_num)
        training_data = SelfPlayDataset(
            turn_files= [f"turn{i}.txt" for i in range(1, 129)],
            game_dir= f"games/game{game_num}"
        )
        train_dataloader = DataLoader(training_data, batch_size=5, shuffle=True)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
        torch.save(model.state_dict(), f"model_params{game_num}.pth")
        torch.save(model.state_dict(), "model_params.pth")
        
train()