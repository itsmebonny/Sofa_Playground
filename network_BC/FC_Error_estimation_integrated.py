
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


import json

import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../simulation_beam'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../npy_GNN'))

from simulation_beam.parameters_2D import p_grid, p_grid_LR




class Net(th.nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.lin1 = th.nn.Linear(input_size, 512)
        self.lin2 = th.nn.Linear(512, 256)
        self.lin3 = th.nn.Linear(256, 128)
        self.lin4 = th.nn.Linear(128, output_size)
        self.relu = th.nn.GELU()
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
    def get_subdirectories(directory):
        return [f.path for f in os.scandir(directory) if f.is_dir()]
    
    def create_data_list(directory):
        samples = DataGraph.get_subdirectories(directory)
        n_samples = len(samples)
        data_list = []
        count = 0
        for i in tqdm(range(n_samples)):
            skip = False
            tmp_dir = samples[i]
            iteration_number = tmp_dir.split('/')[-1].split('_')[-1]
            high_res_displacement = np.load(f"{tmp_dir}/high_res_displacement.npy")
            low_res_displacement = np.load(f"{tmp_dir}/low_res_displacement.npy")
            y = high_res_displacement - low_res_displacement
            low_res_displacement = th.tensor(low_res_displacement, dtype=th.float32).flatten()
            y = th.tensor(y, dtype=th.float32).flatten()
            #load info.json 
            with open(f"{tmp_dir}/info.json") as f:
                info = json.load(f)
                bounding_box = info['bounding_box']
                force_info = info['force_info']
                indices_BC = info['indices_BC']
                x_min = bounding_box['x_min']
                x_max = bounding_box['x_max']
                y_min = bounding_box['y_min']
                y_max = bounding_box['y_max']
                if 'z_min' in bounding_box.keys():
                    z_min = bounding_box['z_min']
                    z_max = bounding_box['z_max']
                    coords = [x_min, x_max, y_min, y_max, z_min, z_max]
                else:
                    coords = [x_min, x_max, y_min, y_max]
                if indices_BC == []:
                    skip = True
                    count += 1
            if not skip:
                boundary_conditions = np.zeros((len(coords) + 4))
                boundary_conditions[:len(coords)] = coords
                boundary_conditions[-4:-1] = force_info['versor']
                boundary_conditions[-1] = force_info['magnitude']
                boundary_conditions = th.tensor(boundary_conditions, dtype=th.float32).flatten()
                node_features = th.cat([low_res_displacement, boundary_conditions])
            else:
                node_features = low_res_displacement
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
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.data_graph = DataGraph(self.data_dir)
        self.nb_nodes = self.data_graph.data_list[0][0].shape[0]
        self.output_size = self.data_graph.data_list[0][1].shape[0]
        self.input_size = self.nb_nodes
        print(f"Input size: {self.input_size}")
        print(f"Output size: {self.output_size}")
        print(f"Number of nodes: {self.nb_nodes}")
        if 'fast_loading' in data_dir:
            self.validation_dir = data_dir
        else:
            self.validation_dir = 'npy_GNN_hf/2025-01-10_10:42:13_validation_250_nodes'
        self.val_data_graph = DataGraph(self.validation_dir)
        self.val_data_list = self.val_data_graph.data_list
        self.data_list = self.data_graph.data_list
        self.loader = DataLoader(self.data_list, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data_list, batch_size=self.batch_size, shuffle=True)
        self.model = Net(self.input_size, self.output_size).to(self.device)
        self.criterion = RMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=15, min_lr=1e-8)

    
    def train(self):
        self.model.train()
        early_stopper = EarlyStopper(patience=60, min_delta=1e-8, lr=self.lr)
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(tqdm(self.loader)):
                x, y = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()
                out = self.model(x)
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
                x, y = data[0].to(self.device), data[1].to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                running_loss += loss.item()
        print(f"Validation Loss: {running_loss/len(self.val_loader)}")
        return running_loss/len(self.val_loader)
    
    def save_model(self, model_dir):
        if not os.path.exists('models_BC'):
            os.mkdir('models_BC')
        model_dir = f'models_BC/{model_dir}.pth'
        th.save(self.model.state_dict(), model_dir)
    
    def load_model(self, model_dir):
        self.model.load_state_dict(th.load(model_dir), strict=False)
        self.model.eval()
    
    def predict(self, data):
        self.model.eval()
        with th.no_grad():
            output = self.model(data)
        return output
    

if __name__ == '__main__':
    data_dir = 'npy_GNN_hf/2025-01-10_01:41:13_training_250_nodes' # CAMBIA NOME MODELLO SE 2D O 3D
    if 'beam' in data_dir:
        print("Beam model")
    else:
        print("2D model")
    trainer = Trainer(data_dir, 32, 0.001, 500)
    trainer.train()
    training_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    #if folder contains 'beam' then save as beam model, otherwise save as GNN model
    if 'beam' in data_dir:
        trainer.save_model(f'model_{training_time}_FC_beam')
    elif '250_nodes' in data_dir:
        trainer.save_model(f'model_{training_time}_FC_250_nodes')
    elif 'hf' in data_dir:
        trainer.save_model(f'model_{training_time}_FC_hf')
    else:
        trainer.save_model(f'model_{training_time}_FC')
    print(f"Model saved as model_{training_time}.pth")
  
    #summary(model, (1, data.input_size))
