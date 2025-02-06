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
import json

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


from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GraphUNet

class Net(nn.Module):
    def __init__(self, nb_nodes, nb_features, message_passing):
        super(Net, self).__init__()
        self.nb_nodes = nb_nodes
        
        # Input processing
        self.input_norm = nn.BatchNorm1d(nb_features)
        
        # GraphUNet - removed add_self_loops parameter
        self.unet = GraphUNet(
            in_channels=nb_features,
            hidden_channels=64,
            out_channels=64,
            depth=message_passing,
            pool_ratios=0.5,
            sum_res=False,
            act=nn.GELU()
        )
        
        # Output projection
        self.final_conv = GCNConv(64, 3)
        #self.final_norm = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_attr):
        batch_size = x.size(0) // self.nb_nodes

        # Network flow with edge attributes
        x = self.input_norm(x)
        x = self.unet(x, edge_index)  # Remove edge_attr parameter
        x = self.dropout(x)
        #x = self.final_norm(x)
        x = self.final_conv(x, edge_index, edge_attr)
        
        return x

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(RMSELoss, self).__init__()
        self.eps = eps
        
    def forward(self, pred, true):
        max_val = th.max(th.abs(true))
        return th.sqrt(th.mean(th.square(pred - true)))/(max_val + self.eps)
    
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
    
    

class DataGraph(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = self.get_subdirectories(data_dir)
        self.n_samples = len(self.samples)
        # Load first sample to get shape info
        self._first_sample = self.__getitem__(0)
    
    @staticmethod
    def get_subdirectories(directory):
        return [f.path for f in os.scandir(directory) if f.is_dir()]
    
    def __len__(self):
        return self.n_samples
    
    @property
    def nb_nodes(self):
        return self._first_sample.x.shape[0]
    
    @property
    def nb_features(self):
        return self._first_sample.x.shape[1]
    
    def __getitem__(self, idx):
        tmp_dir = self.samples[idx]
        
        # Load arrays with memory mapping
        high_res_displacement = np.load(f"{tmp_dir}/high_res_displacement.npy", mmap_mode='r').copy()
        low_res_displacement = np.load(f"{tmp_dir}/low_res_displacement.npy", mmap_mode='r').copy()
        low_res_position = np.load(f"{tmp_dir}/low_res_position.npy", mmap_mode='r').copy()
        edges = np.load(f"{tmp_dir}/edges_low.npy", mmap_mode='r')
        y = high_res_displacement - low_res_displacement
        low_res_displacement = (low_res_displacement - np.mean(low_res_displacement, axis=0))/np.std(low_res_displacement, axis=0)
        low_res_position = (low_res_position - np.min(low_res_position, axis=0))/(np.max(low_res_position, axis=0) - np.min(low_res_position, axis=0))
        
        edge_index = edges[:, :2].T.copy()
        edge_attr = edges[:, 2].copy()
        edge_attr = (edge_attr - np.min(edge_attr))/(np.max(edge_attr) - np.min(edge_attr))

        # Load JSON data
        with open(f"{tmp_dir}/info.json") as f:
            info = json.load(f)
            force_info = info['force_info']
            indices_BC = info['indices_BC']
            #check if indices_hole exists
            if 'indices_circle' in info:
                indices_hole = info['indices_circle']
        
        # Create boundary conditions
        boundary_conditions = np.zeros((low_res_displacement.shape[0], 4))
        if indices_BC:
            boundary_conditions[indices_BC, :3] = force_info['versor']
            boundary_conditions[indices_BC, 3] = force_info['magnitude']
        if 'indices_circle' in info:
            low_res_displacement[indices_hole] = 0


        # Prepare features and target
        node_features = np.hstack((low_res_displacement.copy(), low_res_position.copy(), boundary_conditions))
        
        
        # Create data object
        data = Data(
            x=th.tensor(node_features, dtype=th.float32),
            edge_index=th.tensor(edge_index, dtype=th.long),
            edge_attr=th.tensor(edge_attr, dtype=th.float32),
            y=th.tensor(y, dtype=th.float32)
        )
        
        return data




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
    def __init__(self, data_dir, batch_size, learning_rate, num_epochs, message_passing=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = num_epochs
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.message_passing = message_passing
        
        # Initialize datasets
        self.data_graph = DataGraph(self.data_dir)
        self.nb_nodes = self.data_graph.nb_nodes
        self.nb_features = self.data_graph.nb_features
        
        # Create data loaders with persistent workers
        self.loader = DataLoader(
            self.data_graph, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )
        
        if 'fast_loading' in data_dir:
            self.validation_dir = data_dir
        else:
            self.validation_dir = 'npy_GNN_lego/2025-02-06_23:44:34_validation'
            
        self.val_data_graph = DataGraph(self.validation_dir)
        self.val_loader = DataLoader(
            self.val_data_graph,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )
        self.model = Net(self.nb_nodes, self.nb_features, self.message_passing).to(self.device)
        self.criterion = RMSELoss()
        #ADAM optimizer with tichonov regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=30, min_lr=1e-5)
        self.train_losses = []
        self.val_losses = []
        self.train_mse = []
        self.val_mse = []
        self.train_rmse = []
        self.val_rmse = []
        
    def train(self):
        self.model.train()
        early_stopper = EarlyStopper(patience=40, min_delta=1e-4, lr=self.lr)
        for epoch in range(self.epochs):
            running_loss = 0.0
            running_mse = 0.0
            running_rmse = 0.0
            for i, data in enumerate(tqdm(self.loader)):
                x, edge_index, edge_attr, y = data.x.to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device), data.y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x, edge_index, edge_attr)
                loss = self.criterion(out, y)
                mse = F.mse_loss(out, y)
                rmse = th.sqrt(mse)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_mse += mse.item()
                running_rmse += rmse.item()
                
            epoch_loss = running_loss/len(self.loader)
            epoch_mse = running_mse/len(self.loader)
            epoch_rmse = np.sqrt(epoch_mse)
            self.train_losses.append(epoch_loss)
            self.train_mse.append(epoch_mse)
            self.train_rmse.append(epoch_rmse)
            
            
            
            val_loss, val_mse, val_rmse = self.validate()
            
            self.val_losses.append(val_loss)
            self.val_mse.append(val_mse)
            self.val_rmse.append(val_rmse)
            
            
            if early_stopper.early_stop(val_loss, self.optimizer.param_groups[0]['lr']):
                print("Early stopping")
                break
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Validation Loss: {val_loss}\nCurrent LR: {self.optimizer.param_groups[0]['lr']}\nMetrics:\nMSE: {epoch_mse}, Val MSE: {val_mse}\nRMSE: {epoch_rmse}, Val RMSE: {val_rmse}")
            
        
        
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        running_mse = 0.0
        running_rmse = 0.0
        with th.no_grad():
            for i, data in enumerate(self.val_loader):
                x, edge_index, edge_attr, y = data.x.to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device), data.y.to(self.device)
                out = self.model(x, edge_index, edge_attr)
                loss = self.criterion(out, y)
                mse = F.mse_loss(out, y)
                rmse = th.sqrt(mse)
                running_loss += loss.item()
                running_mse += mse.item()
                running_rmse += rmse.item()
        return running_loss/len(self.val_loader), running_mse/len(self.val_loader), running_rmse/len(self.val_loader)
    
    def save_plots(self, model_dir):
        import matplotlib.pyplot as plt
        
        # Create loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(f'images/loss_{model_dir}.png')
        plt.close()
        
        # Create MSE plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_mse, label='Training MSE')
        plt.plot(self.val_mse, label='Validation MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Training and Validation MSE')
        plt.legend()
        plt.savefig(f'images/mse_{model_dir}.png')
        plt.close()
    
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
        x, edge_index, edge_attr = th.tensor(data.x, dtype=th.float32).to(self.device), th.tensor(data.edge_index, dtype=th.long).to(self.device), th.tensor(data.edge_attr, dtype=th.float32).to(self.device)
        with th.no_grad():
            output = self.model(x, edge_index, edge_attr)
        return output
    

if __name__ == '__main__':
    data_dir = 'npy_GNN_lego/2025-02-06_23:44:34_training'
    message_passing = 5
    trainer = Trainer(data_dir, 32, 0.001, 500, message_passing)
    trainer.train()
    model_name = f"model_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_GNN_passing_{message_passing}"
    if 'beam' in data_dir:
        model_name += '_beam'
    elif '250_nodes' in data_dir:
        model_name += '_250_nodes'
    elif 'hf' in data_dir:
        model_name += '_hf'
    elif 'lego' in data_dir:
        model_name += '_lego'
    trainer.save_model(model_name)
    trainer.save_plots(model_name)
    print(f"Model saved as {model_name}.pth")
  
    #summary(model, (1, data.input_size))