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
    # Dataset class that loads the data from the npy files
    def __init__(self, data_dir, quantity = 'u'):
        self.data_dir = data_dir
        self.normalized = False
        if quantity == 'u' or quantity == 'U' or quantity == 'displacement' or quantity == 'Displacement':
            self.quantity = 'Points'
        elif quantity == 'v' or quantity == 'V' or quantity == 'velocity' or quantity == 'Velocity':
            self.quantity = 'Velocities'
        else:
            raise ValueError("Quantity must be either 'u' or 'v'")
        try:
            self.data_3D = np.load(f'{self.data_dir}/CoarseRes{self.quantity}.npy')
        except FileNotFoundError:
            self.data_3D = np.load(f'{self.data_dir}/CoarseRes{self.quantity}_normalized.npy')
            self.normalized = True
        self.data = self.data_3D.reshape(self.data_3D.shape[0], -1)
        self.input_size = self.data.shape[1]
        try:
            self.labels_3D = np.load(f'{self.data_dir}/HighRes{self.quantity}.npy')
        except FileNotFoundError:
            self.labels_3D = np.load(f'{self.data_dir}/HighRes{self.quantity}_normalized.npy')
        self.labels = self.labels_3D.reshape(self.labels_3D.shape[0], -1)
        self.output_size = self.labels.shape[1]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

class Trainer:
    # Trainer class that trains the model
    def __init__(self, data_dir, batch_size, lr, epochs, dt):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.dt = dt
        self.epochs = epochs
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.train_displacement, self.val_displacement, input_size_u, output_size_u, self.normalized_u = self.load_data('u')
        self.train_velocity, self.val_velocity, input_size_v, output_size_v, self.normalized_v = self.load_data('v')
        print(f"Input size: {input_size_u}, Output size: {output_size_u}")
        self.train_loader_u = DataLoader(self.train_displacement, batch_size=self.batch_size, shuffle=True)
        self.val_loader_u = DataLoader(self.val_displacement, batch_size=self.batch_size, shuffle=False)
        self.train_loader_v = DataLoader(self.train_velocity, batch_size=self.batch_size, shuffle=True)
        self.val_loader_v = DataLoader(self.val_velocity, batch_size=self.batch_size, shuffle=False)
        self.model_u = FullyConnected(input_size_u, output_size_u).to(self.device)
        self.model_v = FullyConnected(input_size_v, output_size_v).to(self.device)
        self.optimizer = optim.Adam([{'params': self.model_u.parameters(), 'lr': self.lr}, {'params': self.model_v.parameters(), 'lr': self.lr*10}])
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=20)
        
    
    def load_data(self, quantity = 'u'):
        data = Data(self.data_dir, quantity)
        train_data, val_data = train_test_split(data, test_size=0.3)
        return train_data, val_data, data.input_size, data.output_size, data.normalized
    
    def train(self):
        self.model_u.train()
        self.model_v.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for item1, item2 in zip(tqdm(self.train_loader_u), self.train_loader_v):
                inputs_u, labels_u = item1
                inputs_v, labels_v = item2
                inputs_u, labels_u = inputs_u.to(self.device), labels_u.to(self.device)
                inputs_v, labels_v = inputs_v.to(self.device), labels_v.to(self.device)
                outputs_u = self.model_u(inputs_u.float())
                outputs_v = self.model_v(inputs_v.float())
                loss1 = self.criterion(outputs_u, labels_u.float())
                loss2 = self.criterion(outputs_v, labels_v.float())
                outputs = outputs_u + outputs_v*self.dt
                loss3 = self.criterion(outputs, labels_u.float())
                self.optimizer.zero_grad()
                loss = loss1 + loss2 + loss3
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {running_loss/len(self.train_loader_u)}")
            val_loss = self.validate()
            if val_loss is not None:
                self.scheduler.step(val_loss)
    
    def validate(self):
        self.model_u.eval()
        self.model_v.eval()
        running_loss = 0.0
        with th.no_grad():
            for item1, item2 in zip(self.val_loader_u, self.val_loader_v):
                inputs_u, labels_u = item1
                inputs_v, labels_v = item2
                inputs_u, labels_u = inputs_u.to(self.device), labels_u.to(self.device)
                inputs_v, labels_v = inputs_v.to(self.device), labels_v.to(self.device)
                outputs_u = self.model_u(inputs_u.float())
                outputs_v = self.model_v(inputs_v.float())
                loss1 = self.criterion(outputs_u, labels_u.float())
                loss2 = self.criterion(outputs_v, labels_v.float())
                outputs = outputs_u + outputs_v*self.dt
                loss3 = self.criterion(outputs, labels_u.float())
                loss = loss1 + loss2 + loss3
                running_loss += loss.item()

        print(f"Validation Loss: {running_loss/len(self.val_loader_u)}")
        return running_loss/len(self.val_loader_u)
    
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

if __name__ == "__main__":
    train = True
    if train:
        data_dir = 'npy/2024-04-25_16:00:00_dynamic_simulation/train'
        trainer = Trainer(data_dir, 32, 0.001, 1000, 0.05)
        trainer.train()
        training_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        trainer.save_model(f'model_{training_time}')
        print(f"Model saved as model_{training_time}.pth")
        # device = th.device("cuda" if th.cuda.is_available() else "cpu") # PyTorch v0.4.0
        # model = FullyConnected(1875, 15000).to(device)

        # summary(model, (1, 1875))
    else:
        data_dir = 'npy/2024-04-24_16:14:43/test'
        model_dir = 'models/model_2024-04-24_16:25:26.pth'
        L2_error = []
        MSE_error = []

        trainer = Trainer(data_dir, 32, 0.001, 1000, 0.005)
        trainer.load_model(model_dir)
        for i, data in enumerate(trainer.val_loader):
            inputs, labels = data
            outputs = trainer.predict(inputs).cpu()
            L2_error.append(th.norm(outputs - labels, p=2).item())
            MSE_error.append(th.norm(outputs - labels, p=2).item()**2)
        print(f"L2 error: {np.mean(L2_error)}")
        print(f"MSE error: {np.mean(MSE_error)}")
        print(f"RMSE error: {np.sqrt(np.mean(MSE_error))}")
        print(f"Max error: {np.max(L2_error)}")
        print(f"Min error: {np.min(L2_error)}")

        # print(f"Predicted: {outputs}")


