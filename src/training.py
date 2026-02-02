import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from augment_data import augment_data
from collections.abc import Callable
from neural_network import NeuralNetwork
from self_play import self_play

class SelfPlayDataset(Dataset):
    def __init__(self, turn_files : list[str], game_dir : str):
        self.turn_files = turn_files
        self.game_dir = game_dir
    
    def __len__(self):
        return len(self.turn_files)
    
    def __getitem__(self, index : int) -> tuple[torch.Tensor, (torch.Tensor, torch.Tensor)]:
        game_path = self.game_dir + "/" + self.turn_files[index]
        state_string = ""
        move_prob = ""
        with open(game_path, 'r') as file:
            for i in range(17):
                state_string+=(next(file))
            while("]" not in move_prob):
                move_prob += next(file)
            
            win = float(next(file))
        
        lines = state_string.strip().split("\n")
        state = []
        for line in lines:
            if "|" not in line: 
                row = line.split(" - ")  
                for i in range(len(row)):
                    if(row[i] == "O"): row[i] = 0
                    elif(row[i] == "B"): row[i] = 1
                    else: row[i] = 2
                state.append(row)
        move_prob = move_prob.strip().strip("[]").split(",")
        for i in range(len(move_prob)):
            move_prob[i] = float(move_prob[i])
        state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0)
        move_prob = torch.tensor(move_prob, dtype=torch.float32)
        win = torch.tensor(win, dtype=torch.float32)
        expected = (move_prob, win)
        return state, expected

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model_params.pth", weights_only=True))
print(model)

learning_rate = 1e-3
batch_size = 32
epochs = 5

def loss_fnc(pred : tuple[int, int], y : tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_p, output_v = pred
    target_p, target_v = y
    value_loss = F.mse_loss(output_v.squeeze(-1), target_v)
    policy_loss = - torch.mean(torch.sum(target_p * F.log_softmax(output_p, dim=1), dim=1))
    loss = value_loss + policy_loss
    return loss, value_loss, policy_loss

loss_fn = loss_fnc

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)

def train_loop(dataloader : DataLoader, model : NeuralNetwork, 
               loss_fn : Callable[[tuple[int, int], tuple[int, int]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
               optimizer : torch.optim) -> None:
    size = len(dataloader.dataset)
    model.train()

    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device)
        print("Input device:", X.device)
        target_p, target_v = y
        target_p = target_p.to(device)
        target_v = target_v.to(device)
        y = (target_p, target_v)
        output = model(X)
        loss, v_loss, p_loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        total_norm = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item()

        print("grad norm:", total_norm)
        optimizer.step()
        if batch % 4 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"value loss: {v_loss:>7f}")
            print(f"policy loss: {p_loss:>7f}")

torch.autograd.set_detect_anomaly(True)
def train() -> None:
    for game_num in range(1, 50):
        model = NeuralNetwork()
        model.load_state_dict(torch.load("model_params.pth", weights_only=True))
        model.eval()
        model = model.to(device)
        num_moves = self_play(game_num, model)
        os.makedirs(f"games/game{game_num}r")
        os.makedirs(f"games/game{game_num}rr")
        os.makedirs(f"games/game{game_num}rrr")
        os.makedirs(f"games/game{game_num}hf")
        os.makedirs(f"games/game{game_num}vf")
        augment_data(game_num, num_moves)
        turn_files = [f"turn{i}.txt" for i in range(1, num_moves)]
        training_data = SelfPlayDataset(
            turn_files= turn_files,
            game_dir= f"games/game{game_num}"
        )
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)        

        training_data = SelfPlayDataset(
            turn_files= turn_files,
            game_dir= f"games/game{game_num}r"
        )
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)

        training_data = SelfPlayDataset(
            turn_files= turn_files,
            game_dir= f"games/game{game_num}rr"
        )
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)

        training_data = SelfPlayDataset(
            turn_files= turn_files,
            game_dir= f"games/game{game_num}rrr"
        )
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)

        training_data = SelfPlayDataset(
            turn_files= turn_files,
            game_dir= f"games/game{game_num}hf"
        )
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)

        training_data = SelfPlayDataset(
            turn_files= turn_files,
            game_dir= f"games/game{game_num}vf"
        )
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
        torch.save(model.state_dict(), "model_params.pth")
train()