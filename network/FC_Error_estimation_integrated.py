
import numpy as np 
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
from datetime import datetime

# playground for GNN data processing

import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch as th


import os

import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../simulation_beam'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../npy_GNN'))

from simulation_beam.parameters_2D import p_grid, p_grid_LR


from torchsummary import summary

from torch_geometric.loader import DataLoader



class Net(th.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = th.nn.Linear(225, 512)
        self.lin2 = th.nn.Linear(512, 256)
        self.lin3 = th.nn.Linear(256, 128)
        self.lin4 = th.nn.Linear(128, 225)
        self.relu = th.nn.ReLU()
        self.dropout = th.nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin4(x)
        return x
    
    

    

class RMSELoss(nn.Module):
    # Custom loss function that computes the mean squared error between the predicted and true values and normalizes it by the true value
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, true):
        loss = th.mean(th.square(pred - true)) 
        return th.sqrt(loss)
    
class MixedLoss(nn.Module):
    # Custom loss function that computes the weighted sum between the root mean squared error and the maximum absolute error
    def __init__(self):
        super(MixedLoss, self).__init__()

    def forward(self, pred, true):
        # select the indices where the absolute value is greater than 1
        mask = th.abs(true) > 1
        if th.sum(mask) > 200:
            loss1 = th.sqrt(th.mean(th.square(pred[mask] - true[mask])))
            loss2 = th.sqrt(th.mean(th.square(pred[~mask] - true[~mask])))
            return 0.7*loss1 + 0.3*loss2
        else:
            loss1 = th.sqrt(th.mean(th.square(pred - true)))
            loss2 = th.max(th.abs(pred - true))
            return 0.7*loss1 + 0.3*loss2
    
    
# create class that loads the data from the npy files and manipulates them to prepare them for training for pytorch geometric

class DataGraph(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = DataGraph.create_data_list(data_dir)

        
    def __cat_dim__(self, key, value, *args, **kwargs):
        return None 
    
        # load the data
    def get_filenames(directory):
        filenames = []
        for file in os.listdir(f"{directory}"):
            if file[-4:] == ".npy":
                filenames.append(file)
        filenames.sort()
        return filenames
    
    def get_filenames_no_time(directory):
        filenames = []
        for file in os.listdir(f"{directory}"):
            if file[-4:] == ".npy":
                filenames.append(file)
        filenames.sort()
        for i in range(len(filenames)):
            filenames[i] = filenames[i].split("_")[0] + ".npy"
        return filenames
    
    def create_data_list(directory):
        names = DataGraph.get_filenames(directory)
        print(f"Number of FILES: {len(names)}")
        names_no_time = DataGraph.get_filenames_no_time(directory)
        types = len(set(names_no_time))
        print(f"Number of TYPES: {types}")
        samples = len(names) // types
        data_list = []
        for i in range(samples):
            high_res_displacement = np.load(f"{directory}/{names[samples*3+i]}")
            low_res_displacement = np.load(f"{directory}/{names[i]}")
            # timestep = names[i].split("_")[6]
            # timestep = timestep.split(".")[0]
            # timestep_nodes = (int(timestep) % 1000) * np.ones((75, 1))
            high_res_velocity = np.load(f"{directory}/{names[samples*4+i]}")
            low_res_velocity = np.load(f"{directory}/{names[samples*5+i]}")
            node_features = low_res_displacement #np.hstack((low_res_displacement))#, low_res_velocity, timestep_nodes))
            #node_features = low_res_velocity
            # edge_index = np.load(f"{directory}/{names[samples*2+i]}")[:, :2].T
            # edge_attr = np.load(f"{directory}/{names[samples*2+i]}")[:, 2]
            y = high_res_displacement - low_res_displacement
            y = y.flatten()
            data_list.append([node_features, y])
        return data_list

    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, lr = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.lr = lr

    def early_stop(self, validation_loss, learning_rate):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif learning_rate != self.lr:
            self.lr = learning_rate
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print(f"Counter: {self.counter}")
            print(f"Diff: {validation_loss - self.min_validation_loss}")
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    # Trainer class that trains the model
    def __init__(self, data_dir, batch_size, lr, epochs):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        foo = 1
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.data_graph = DataGraph(self.data_dir)
        self.validation_dir = 'npy_GNN/2024-10-24_17:46:55_estimation'
        self.val_data_graph = DataGraph(self.validation_dir)
        self.val_data_list = self.val_data_graph.data_list
        self.data_list = self.data_graph.data_list
        self.loader = DataLoader(self.data_list, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data_list, batch_size=self.batch_size, shuffle=True)
        self.model = Net().to(self.device)
        self.criterion = RMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=15, min_lr=1e-8)

    
    def train(self):
        self.model.train()
        early_stopper = EarlyStopper(patience=60, min_delta=1e-8, lr=self.lr)
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(tqdm(self.loader)):
                x, edge_index, edge_attr, y = data.x.to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device), data.y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x, edge_index, edge_attr)
                y = y.view(-1, 225)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(self.loader)}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
            val_loss = self.validate()
            if val_loss is not None:
                if early_stopper.early_stop(val_loss, self.optimizer.param_groups[0]['lr']):
                    print("Early stopping")
                    break
                self.scheduler.step(val_loss)
        
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        with th.no_grad():
            for i, data in enumerate(self.val_loader):
                x, edge_index, edge_attr, y = data.x.to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device), data.y.to(self.device)
                out = self.model(x, edge_index, edge_attr)
                y = y.view(-1, 225)
                loss = self.criterion(out, y)
                running_loss += loss.item()
        print(f"Validation Loss: {running_loss/len(self.val_loader)}")
        return running_loss/len(self.val_loader)
    
    def save_model(self, model_dir):
        if not os.path.exists('models'):
            os.mkdir('models')
        model_dir = f'models_GNN/{model_dir}.pth'
        th.save(self.model.state_dict(), model_dir)
    
    def load_model(self, model_dir):
        self.model.load_state_dict(th.load(model_dir), strict=False)
        self.model.eval()
    
    def predict(self, data):
        self.model.eval()
        x, edge_index, edge_attr = th.tensor(data.x, dtype=th.float32).to(self.device), th.tensor(data.edge_index, dtype=th.long).to(self.device), th.tensor(data.edge_attr, dtype=th.float32).to(self.device)
        with th.no_grad():
            output = self.model(x, edge_index, edge_attr)
        return output
    

if __name__ == '__main__':
    data_dir = 'npy_GNN/2024-10-29_18:03:06_estimation'
    trainer = Trainer(data_dir, 16, 0.001, 500)
    trainer.train()
    training_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    trainer.save_model(f'model_{training_time}_GNN')
    print(f"Model saved as model_{training_time}.pth")
  
    #summary(model, (1, data.input_size))
