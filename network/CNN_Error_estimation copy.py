from cv2 import normalize
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
        self.dropout = nn.Dropout(0.25)
        self.fc4 = nn.Linear(1024, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x)) # use sin if you have periodicity in the data
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
    
class Convolution2D(nn.Module):
    # Convolutional neural network with 2D convolutional layers with input shape [None, 2500, 3, 2]
    def __init__(self, input_shape, output_size):
        super(Convolution2D, self).__init__()
        self.conv1 = nn.Conv2d(2500, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(12*32, 128)
        self.fc2 = nn.Linear(128, output_size)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

    

class RelativeMSELoss(nn.Module):
    # Custom loss function that computes the mean squared error between the predicted and true values and normalizes it by the true value
    def __init__(self):
        super(RelativeMSELoss, self).__init__()

    def forward(self, pred, true):
        loss = th.mean(th.square(pred - true)) 
        normalizer = th.mean(th.square(true))
        return loss/normalizer
    
class MixedLoss(nn.Module):
    # Custom loss function that computes the weighted sum between the root mean squared error and the maximum absolute error
    def __init__(self):
        super(MixedLoss, self).__init__()

    def forward(self, pred, true):
        loss1 = th.sqrt(th.mean(th.square(pred - true)))
        loss2 = th.max(th.abs(pred - true))
        weight1 = 0.8
        weight2 = 0.2
        return weight1*loss1 + weight2*loss2
    

    
    

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

        # count nb of nodes 
        nb_nodes = self.coarse_3D.shape[1]
        # load the position of the nodes
        #split directory to get the number of nodes
        parent_directory = data_dir.split('/')[0]
        positions = np.load(f'{parent_directory}/{nb_nodes}_nodes/CoarseResPoints.npy')

        positions_repeated = np.repeat(positions[np.newaxis, :, :], self.coarse_3D.shape[0], axis=0)
        

        self.data = np.concatenate((self.coarse_3D[:,:,:,np.newaxis], positions_repeated[:,:,:,np.newaxis]), axis=-1)
        self.labels = self.high_3D - self.coarse_3D

        self.labels = self.labels.reshape(self.labels.shape[0], -1)
        
        self.output_size = self.labels.shape[1]
        self.input_size = self.data.shape[1]
        self.input_shape = self.data.shape
        print(f"Input shape: {self.input_shape}")


    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # add noise to the data
        noise = np.random.normal(0, 0.1, self.data[idx].shape)
        #return self.data[idx] + noise, self.labels[idx]
        return self.data[idx], self.labels[idx]
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
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
        self.train_data, self.val_data, input_size, output_size, self.normalized = self.load_data()
        print(f"Input size: {input_size}, Output size: {output_size}")
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        self.model = Convolution2D(input_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.criterion = MixedLoss()
        #self.criterion = RelativeMSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-8)
        
    
    def load_data(self):
        data = Data(self.data_dir)
        train_data, val_data = train_test_split(data, test_size=0.2)
        return train_data, val_data, data.input_shape, data.output_size, data.normalized
    
    def train(self):
        self.model.train()
        early_stopper = EarlyStopper(patience=40, min_delta=1e-8)
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
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
            val_loss = self.validate()
            if val_loss is not None:
                if early_stopper.early_stop(val_loss):
                    print("Early stopping")
                    break
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
        return running_loss/len(self.val_loader)
    
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
    data_dir = 'npy_beam/2024-07-23_09:23:48_estimation/train'
    data = Data(data_dir)
    model = FullyConnected(data.input_size, data.output_size)
    trainer = Trainer(data_dir, 32, 0.001, 1000)
    trainer.train()
    training_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    trainer.save_model(f'model_{training_time}_beam')
    print(f"Model saved as model_{training_time}.pth")
  
    #summary(model, (1, data.input_size))