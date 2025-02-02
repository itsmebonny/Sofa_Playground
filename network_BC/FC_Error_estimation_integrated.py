
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
    
# class Net(nn.Module): #DEBUG
#     def __init__(self, input_size, output_size):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size * 5042, output_size)  # Flatten input
        
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten: [batch_size, nodes * features]
#         return self.fc1(x)

    

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, true):
        # Reshape tensors to match dimensions
        pred = pred.view(pred.shape[0], -1)  # (batch_size, nb_nodes * 3)
        true = true.view(true.shape[0], -1)  # (batch_size, nb_nodes * 3)
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
        return self._first_sample.shape[0]
    
    @property
    def nb_features(self):
        return self._first_sample.shape[1]
    
    def __getitem__(self, idx):
        tmp_dir = self.samples[idx]
        high_res_displacement = np.load(f"{tmp_dir}/high_res_displacement.npy", mmap_mode='r')
        low_res_displacement = np.load(f"{tmp_dir}/low_res_displacement.npy", mmap_mode='r')

        with open(f"{tmp_dir}/info.json") as f:
            info = json.load(f)
            force_info = info['force_info']
            indices_BC = info['indices_BC']
            bounding_box = info['bounding_box']
        
            # Create boundary conditions with 10 columns: 
            # [force_versor_x,y,z, force_magnitude, bbox_min_x,y,z, bbox_max_x,y,z]
            boundary_conditions = np.zeros((1, 10))
            boundary_conditions[0, :3] = force_info['versor']
            boundary_conditions[0, 3] = force_info['magnitude']
            boundary_conditions[0, 4] = bounding_box['x_min']
            boundary_conditions[0, 5] = bounding_box['y_min']
            boundary_conditions[0, 6] = bounding_box['z_min']
            boundary_conditions[0, 7] = bounding_box['x_max']
            boundary_conditions[0, 8] = bounding_box['y_max']
            boundary_conditions[0, 9] = bounding_box['z_max']

            #print shapes
       
            target = high_res_displacement - low_res_displacement
            high_res_displacement = high_res_displacement.flatten()
            features = np.zeros((high_res_displacement.shape[0]+boundary_conditions.shape[1], 1))
            features[:high_res_displacement.shape[0], 0] = high_res_displacement
            features[high_res_displacement.shape[0]:, 0] = boundary_conditions.flatten()
            #remove last axis of features
            features = np.squeeze(features)
            

            return th.tensor(features, dtype=th.float32), th.tensor(target.flatten(), dtype=th.float32)
    

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
    def __init__(self, data_dir, batch_size, lr, epochs):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        # Create datasets
        self.data_graph = DataGraph(self.data_dir)
        
        # Get dimensions from first sample
        first_features, first_target = self.data_graph[0]
        self.input_size = first_features.shape[0]  # Features per node
        self.output_size = first_target.shape[0]   # Total output size
        self.nb_nodes = first_features.shape[0]    # Number of nodes
        
        print(f"Input size: {self.input_size}")
        print(f"Output size: {self.output_size}")
        # Initialize metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_mse = []
        self.val_mse = []
        
        # Initialize epoch metrics
        self.train_losses_epoch = []
        self.val_losses_epoch = []
        self.train_mse_epoch = []
        self.val_mse_epoch = []
        
        # Setup validation
        if 'fast_loading' in data_dir:
            self.validation_dir = data_dir
        else:
            self.validation_dir = 'npy_GNN_lego/2025-01-31_09:30:54_validation_10k'
            
        self.val_data_graph = DataGraph(self.validation_dir)
        
        # Create dataloaders
        self.loader = DataLoader(
            self.data_graph,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_data_graph,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )
        
        # Setup model and training components
        self.model = Net(self.input_size, self.output_size).to(self.device)
        self.criterion = RMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=15, min_lr=1e-5
        )
        
        # Initialize metric storage
        self.train_losses_epoch = []
        self.val_losses_epoch = []
        self.train_mse_epoch = []
        self.val_mse_epoch = []

    
    def train(self):
        self.model.train()
        early_stopper = EarlyStopper(patience=60, min_delta=1e-8, lr=self.lr)
    
        for epoch in range(self.epochs):
            running_loss = 0.0
            running_mse = 0.0
            
            for batch_idx, (X, y) in enumerate(tqdm(self.loader)):
                X = X.to(self.device)
                y = y.to(self.device)
                
         
                
                self.optimizer.zero_grad()
                out = self.model(X)
                
         
                
                # Reshape output to match target
                out = out.view(y.shape)
                loss = self.criterion(out, y)
                mse = F.mse_loss(out, y)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                running_mse += mse.item()
                
            # Store epoch metrics
            epoch_loss = running_loss/len(self.loader)
            epoch_mse = running_mse/len(self.loader)
            self.train_losses_epoch.append(epoch_loss)
            self.train_mse_epoch.append(epoch_mse)
            
            # Validation step
            val_loss, val_mse = self.validate()
            self.val_losses_epoch.append(val_loss)
            self.val_mse_epoch.append(val_mse)
            
            print(f'Epoch {epoch+1}: Train Loss={epoch_loss:.6f}, Val Loss={val_loss:.6f}')
            
            if early_stopper.early_stop(val_loss, self.optimizer.param_groups[0]['lr']):
                print("Early stopping")
                break
            
            self.scheduler.step(val_loss)
            
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        running_mse = 0.0
        
        with th.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                mse = F.mse_loss(out, y)
                
                running_loss += loss.item()
                running_mse += mse.item()
        
        val_loss = running_loss/len(self.val_loader)
        val_mse = running_mse/len(self.val_loader)
        
        self.val_losses.append(val_loss)
        self.val_mse.append(val_mse)
        
        return val_loss, val_mse
    
    def save_plots(self, model_dir):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(f'images/loss_{model_dir}.png')
        plt.close()
        
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
        with th.no_grad():
            output = self.model(data)
        return output
    

if __name__ == '__main__':
    data_dir = 'npy_GNN_lego/2025-01-31_01:12:47_training_10k'
    trainer = Trainer(data_dir, 32, 0.001, 500)
    trainer.train()
    model_name = f"model_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_FC"
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
