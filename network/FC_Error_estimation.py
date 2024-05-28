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

from torchsummary import summary

class FullyConnected(nn.Module):
    # Fully connected neural network with 4 hidden layers
    def __init__(self, input_size, output_size):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.2)
        self.fc4 = nn.Linear(1024, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x)) # use sin if you have periodicity in the data
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
    

class Data(Dataset):
    # Dataset class that loads the data from the npy files and manipulates them to prepare them for training

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.normalized = False
        try:
            self.coarse_3D = np.load(f'{self.data_dir}/CoarseResPoints.npy')
        except FileNotFoundError:
            self.coarse_3D = np.load(f'{self.data_dir}/CoarseResPoints_normalized.npy')
            self.normalized = True

        try:
            self.high_3D = np.load(f'{self.data_dir}/HighResPoints.npy')
        except FileNotFoundError:
            self.high_3D = np.load(f'{self.data_dir}/HighResPoints_normalized.npy')

        self.data = self.coarse_3D.reshape(self.coarse_3D.shape[0], -1)
        self.labels = self.high_3D - self.coarse_3D
        self.labels = self.labels.reshape(self.labels.shape[0], -1)
        
        self.output_size = self.labels.shape[1]
        self.input_size = self.data.shape[1]



    def pooling_3D(self, displacement, low_resh_shape=None):
        if low_resh_shape is None:
            low_resh_shape = displacement.shape[1]//2, displacement.shape[2]//2, displacement.shape[3]
        y, x, z = low_resh_shape
        pool = np.zeros((displacement.shape[0], y, x, z))

        for i in range(displacement.shape[0]):
            for j in range(y):
                for k in range(x):
                    # check if the point is on the boundary
                    if k == 0:
                        pool[i, j, k] = displacement[i, 2*j, 2*k]
                    else:
                        # this must be fixed since it assumes that the pooling kernel is 2x2 and i'm not that a good computer scientist but works on my toy example
                        pool[i, j, k] = np.mean(displacement[i, 2*j:2*j+2, 2*k:2*k+2], axis=(0, 1))
        
        return pool
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # add noise to the data
        noise = np.random.normal(0, 0.1, self.data[idx].shape)
        return self.data[idx] + noise, self.labels[idx]
        #return self.data[idx], self.labels[idx]

class Trainer:
    # Trainer class that trains the model
    def __init__(self, data_dir, batch_size, lr, epochs):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.train_data, self.val_data, input_size, output_size, self.normalized = self.load_data()
        print(f"Input size: {input_size}, Output size: {output_size}")
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        self.model = FullyConnected(input_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=20, verbose=True)
        
    
    def load_data(self):
        data = Data(self.data_dir)
        train_data, val_data = train_test_split(data, test_size=0.3)
        return train_data, val_data, data.input_size, data.output_size, data.normalized
    
    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(tqdm(self.train_loader)):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs, labels.float())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(self.train_loader)}")
            val_loss = self.validate()
            if val_loss is not None:
                self.scheduler.step(val_loss)
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        with th.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs, labels.float())
                running_loss += loss.item()
        print(f"Validation Loss: {running_loss/len(self.val_loader)}")
    
    def save_model(self, model_dir):
        if not os.path.exists('models'):
            os.mkdir('models')
        if self.normalized:
            model_dir = f'models/{model_dir}_normalized.pth'
        else:
            model_dir = f'models/{model_dir}.pth'
        th.save(self.model.state_dict(), model_dir)
    
    def load_model(self, model_dir):
        self.model.load_state_dict(th.load(model_dir))
        self.model.eval()
    
    def predict(self, input_data):
        self.model.eval()
        with th.no_grad():
            input_data = th.tensor(input_data).to(self.device)
            output = self.model(input_data.float())
        return output
    

if __name__ == '__main__':
    data_dir = 'npy_gmsh/2024-05-28_11:10:16_estimation_efficient_183nodes/train'
    data = Data(data_dir)
    model = FullyConnected(data.input_size, data.output_size)
    trainer = Trainer(data_dir, 32, 0.001, 1000)
    trainer.train()
    training_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    trainer.save_model(f'model_{training_time}_noisy')
    print(f"Model saved as model_{training_time}.pth")
  
    #summary(model, (1, data.input_size))
