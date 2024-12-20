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
    


    

class RelativeMSELoss(nn.Module):
    # Custom loss function that computes the mean squared error between the predicted and true values and normalizes it by the true value
    def __init__(self):
        super(RelativeMSELoss, self).__init__()

    def forward(self, pred, true):
        loss = th.mean(th.square(pred - true)) 
        normalizer = th.mean(th.square(true))
        return loss/normalizer
    
class RMSELoss(nn.Module):
    # Custom loss function that computes the root mean squared error between the predicted and true values
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, true):
        return th.sqrt(th.mean(th.square(pred - true)))
    
class MixedLoss(nn.Module):
    # Custom loss function that computes the weighted sum between the root mean squared error and the maximum absolute error
    def __init__(self):
        super(MixedLoss, self).__init__()

    def forward(self, pred, true):
        # select the indices where the absolute value is greater than 1
        mask = th.abs(true) > 0.5
        if th.sum(mask) > 50:
            loss1 = th.sqrt(th.mean(th.square(pred[mask] - true[mask])))
            loss2 = th.sqrt(th.mean(th.square(pred[~mask] - true[~mask])))
            return 0.7*loss1 + 0.3*loss2
        else:
            loss1 = th.sqrt(th.mean(th.square(pred - true)))
            loss2 = th.max(th.abs(pred - true))
            return 0.7*loss1 + 0.3*loss2
        
class ScaleInvariantLoss(nn.Module):
    # Custom loss function that is scale invariant
    def __init__(self, slope=1e-5):
        super(ScaleInvariantLoss, self).__init__()
        self.slope = slope

    def symlog(self, x, slope=1e-5):
        return th.sign(x) * th.log10(th.tensor(1, dtype=th.float32)+ th.abs(x)/th.tensor(slope, dtype=th.float32))
    
    def forward(self,pred, true):
    # Compute the differences di
        d = self.symlog(pred) - self.symlog(true)
        
        # Calculate the first term: (1/n) * sum(di^2)
        term1 = th.mean(th.square(d))
        
        # # Calculate the second term: (1/n^2) * (sum(di))^2
        # term2 = th.square(th.sum(d)) / th.square(len(d))

        # # Compute the final metric D(pred, true)
        # D = 0.7*term1 - 0.3*term2
        
        return term1
    

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

        self.max_data = np.max(self.coarse_3D) #, axis=2)
        self.min_data = np.min(self.coarse_3D) #, axis=2)
        self.data = (self.coarse_3D - self.min_data)/(self.max_data - self.min_data)
        self.data = self.data.reshape(self.data.shape[0], -1)
        print(f"Data shape: {self.data.shape}")
        self.labels = self.high_3D - self.coarse_3D
        self.labels = self.labels.reshape(self.labels.shape[0], -1)
        #find the minimum non-zero value in the labels
        self.min_labels = np.min(np.abs(self.labels[np.nonzero(self.labels)]))
        print(f"Min labels: {self.min_labels}")
        print(f"Max labels: {np.max(self.labels)}")
        
        self.output_size = self.labels.shape[1]
        self.input_size = self.data.shape[1]

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # add noise to the data
        noise = np.random.normal(0, 0.1, self.data[idx].shape)
        #return self.data[idx] + noise, self.labels[idx]
        return self.data[idx], self.labels[idx]
    
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
        self.train_data, self.val_data, input_size, output_size, self.min_data, self.max_data, self.min_labels = self.load_data()
        
        # self.min_labels = 1e-10
        print(f"Input size: {input_size}, Output size: {output_size}")
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        self.model = FullyConnected(input_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.criterion = ScaleInvariantLoss(slope=self.min_labels)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-8)
        
    
    def load_data(self):
        data = Data(self.data_dir)
        train_data, val_data = train_test_split(data, test_size=0.2)
        return train_data, val_data, data.input_size, data.output_size, data.min_data, data.max_data, data.min_labels
    
    def train(self):
        self.model.train()
        early_stopper = EarlyStopper(patience=40, min_delta=1e-9, lr=self.lr)
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
                if early_stopper.early_stop(val_loss, self.optimizer.param_groups[0]['lr']):
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
        else:
            model_dir = f'models/{model_dir}.pth'
        th.save(self.model.state_dict(), model_dir)
    
    def load_model(self, model_dir):
        self.model.load_state_dict(th.load(model_dir))
        self.model.eval()
    
    def predict(self, input_data):
        self.model.eval()
        with th.no_grad():
            input_data = (input_data - self.min_data)/(self.max_data - self.min_data)
            input_data = th.tensor(input_data).to(self.device)
            output = self.model(input_data.float())
            #output = th.sgn(output) * th.tensor(self.min_labels, dtype=th.float32) * (-th.tensor(1, dtype=th.float32)+th.pow(th.tensor(10, dtype=th.float32), th.abs(output)))
        return output
    

if __name__ == '__main__':
    data_dir = 'npy_beam/2024-08-07_16:39:32_estimation/train'

    trainer = Trainer(data_dir, 32, 0.001, 100)
    trainer.train()
    training_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if data_dir.split('/')[-2].split('_')[-1] == 'estimation':
        training_time = training_time + '_estimation'
    elif data_dir.split('/')[-2].split('_')[-1] == 'symmetric':
        training_time = training_time + '_symmetric'
    trainer.save_model(f'model_{training_time}_symlog')
    print(f"Model saved as model_{training_time}.pth")
  
    #summary(model, (1, data.input_size))
