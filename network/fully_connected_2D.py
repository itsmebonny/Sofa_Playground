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


class FullyConnected2D(nn.Module):
    # Fully connected neural network with 4 hidden layers
    def __init__(self, input_size, output_size):
        super(FullyConnected2D, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x)) # use sin if you have periodicity in the data
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x







class Data(Dataset):
    # Dataset class that loads the data from the npy files
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.normalized = False
        try:
            self.data_2D = np.load(f'{self.data_dir}/CoarseResPoints.npy')
        except FileNotFoundError:
            self.data_2D = np.load(f'{self.data_dir}/CoarseResPoints_normalized.npy')
            self.normalized = True
        self.data = self.data_2D.reshape(self.data_2D.shape[0], -1)
        self.input_size = self.data.shape[1]
        try:
            self.labels_2D = np.load(f'{self.data_dir}/HighResPoints.npy')
        except FileNotFoundError:
            self.labels_2D = np.load(f'{self.data_dir}/HighResPoints_normalized.npy')
        self.labels = self.labels_2D.reshape(self.labels_2D.shape[0], -1)
        self.output_size = self.labels.shape[1]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

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
        self.model = FullyConnected2D(input_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=100)
        
    
    def load_data(self):
        data = Data(self.data_dir)
        train_data, val_data = train_test_split(data, test_size=0.2)
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
            print(f"Epoch {epoch+1}, Loss: {loss}")
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




if __name__ == "__main__":
    train = 0
    if train:
        data_dir = 'npy/2024-04-22_15:08:52/train'
        trainer = Trainer(data_dir, 32, 0.0001, 500)
        trainer.train()
        training_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        trainer.save_model(f'model_{training_time}')
        print(f"Model saved as model_{training_time}.pth")
        # device = th.device("cuda" if th.cuda.is_available() else "cpu") # PyTorch v0.4.0
        # model = FullyConnected(1875, 15000).to(device)

        # summary(model, (1, 1875))
    else:
    # make some prediction on the test set
        test_data = 'npy/2024-04-22_15:08:52/test'
        model_dir = 'models/model_2024-04-23_14:48:09.pth'
        predictor = Trainer(test_data, 32, 0.0001, 500)
        predictor.load_model(model_dir)

        # compute statistics on the test set
        test_data = Data(test_data)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        criterion = nn.MSELoss()
        predictor.model.eval()
        running_loss = 0.0
        with th.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(predictor.device), labels.to(predictor.device)
                outputs = predictor.model(inputs.float())
                loss = criterion(outputs, labels.float())
                running_loss += loss.item()
        print(f"Test Loss: {running_loss/len(test_loader)}")
        # make a prediction
        input_data = test_data[0][0]
        output = predictor.predict(input_data).cpu().numpy()
        #print(f"Prediction: {output}")
        #print(f"True value: {test_data[0][1]}")
        print(f"Prediction error: {np.linalg.norm(output - test_data[0][1])}")
        print(f"Prediction error: {np.linalg.norm(output - test_data[0][1])/np.linalg.norm(test_data[0][1])}")
        print(f"Prediction error: {np.linalg.norm(output - test_data[0][1])/np.linalg.norm(test_data[0][1]) * 100}%")

