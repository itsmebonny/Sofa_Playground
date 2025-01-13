
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



class Net(th.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 1)  # Output single value per node
        
    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
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
            edge_index = np.load(f"{tmp_dir}/edges_low.npy")[:, :2].T
            edge_attr = np.load(f"{tmp_dir}/edges_low.npy")[:, 2]

            #load info.json 
            with open(f"{tmp_dir}/info.json") as f:
                info = json.load(f)
                bounding_box = info['bounding_box']
                force_info = info['force_info']
                indices_BC = info['indices_BC']
                if indices_BC == []:
                    skip = True
                    count += 1
            if not skip:
                boundary_conditions = np.zeros((low_res_displacement.shape[0], 4))
                boundary_conditions[indices_BC, :3] = force_info['versor']
                boundary_conditions[indices_BC, 3] = force_info['magnitude']


                node_features = np.hstack((low_res_displacement, boundary_conditions))
                y = high_res_displacement - low_res_displacement
                y = y.flatten()
                data = Data(x=th.tensor(node_features, dtype=th.float32), edge_index=th.tensor(edge_index, dtype=th.long), edge_attr=th.tensor(edge_attr, dtype=th.float32), y=th.tensor(y, dtype=th.float32))
                data_list.append(data)
        print(f"Skipped {count} samples, this dataset has a percentage of {count/n_samples*100}% of skipped samples")
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
    def __init__(self, data_dir, batch_size, lr, epochs):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.validation_dir = "npy_GNN_hf/2025-01-10_10:42:13_validation_250_nodes"
        
        # Load data and get dimensions
        self.data_graph = DataGraph(self.data_dir)
        self.nb_nodes = self.data_graph.data_list[0].x.shape[0]
        self.nb_features = self.data_graph.data_list[0].x.shape[1]

        print(f"Initial number of nodes: {self.nb_nodes}")
        print(f"Number of features: {self.nb_features}")
        
       # Process data and validate indices
        max_nodes = 0
        for data in self.data_graph.data_list:
            max_nodes = max(max_nodes, data.x.shape[0])
            # Ensure edge indices are within bounds
            data.edge_index = th.clamp(data.edge_index, 0, data.x.shape[0] - 1)
            # Add batch information properly
            data.batch = th.zeros(data.x.shape[0], dtype=th.long)
            data.ptr = th.tensor([0, data.x.shape[0]])

        self.nb_nodes = max_nodes
        print(f"Max nodes across dataset: {self.nb_nodes}")

        # Create loader with custom collate function
        def custom_collate(data_list):
            batch = Batch.from_data_list(data_list)
            # Verify edge indices after batching
            batch.edge_index = th.clamp(batch.edge_index, 0, batch.x.shape[0] - 1)
            return batch

        self.loader = DataLoader(
            self.data_graph.data_list,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate
        )

        # Apply same processing to validation data
        for data in self.val_data_graph.data_list:
            data.edge_index = th.clamp(data.edge_index, 0, data.x.shape[0] - 1)
            data.batch = th.zeros(data.x.shape[0], dtype=th.long)
            data.ptr = th.tensor([0, data.x.shape[0]])

        self.val_loader = DataLoader(
            self.val_data_graph.data_list,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate
        )

        # Initialize model
        self.model = Net(self.nb_features, hidden_channels=64).to(self.device)
        self.criterion = RMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=15, min_lr=1e-8)

    
    def train(self):
        self.model.train()
        early_stopper = EarlyStopper(patience=60, min_delta=1e-8, lr=self.lr)
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, batch in enumerate(tqdm(self.loader)):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                # Process batch properly
                out = self.model(batch.x, batch.edge_index, batch.edge_attr)
                y = batch.y.view(batch.num_graphs, -1)  # Reshape target properly
                
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
                y = y.view(-1, self.nb_nodes*3)
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
        x, edge_index, edge_attr = th.tensor(data.x, dtype=th.float32).to(self.device), th.tensor(data.edge_index, dtype=th.long).to(self.device), th.tensor(data.edge_attr, dtype=th.float32).to(self.device)
        with th.no_grad():
            output = self.model(x, edge_index, edge_attr)
        return output
    

if __name__ == '__main__':
    data_dir = 'npy_GNN_hf/2025-01-10_01:41:13_training_250_nodes'
    trainer = Trainer(data_dir, 32, 0.001, 500)
    trainer.train()
    training_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if 'beam' in data_dir:
        trainer.save_model(f'model_{training_time}_GNN_beam')
    elif '250_nodes' in data_dir:
        trainer.save_model(f'model_{training_time}_GNN_250_nodes')
    elif 'hf' in data_dir:
        trainer.save_model(f'model_{training_time}_FC_hf')
    else:
        trainer.save_model(f'model_{training_time}_FC')
    print(f"Model saved as model_{training_time}.pth")
  
    #summary(model, (1, data.input_size))
