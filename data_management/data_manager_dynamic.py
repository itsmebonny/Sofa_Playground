from cv2 import norm
import numpy as np
import os
import shutil






class DataManager:

    def __init__(self, data_dir = None) -> None:
        # Creates a DataManager object that takes as input the directory where the data is stored
        # If no directory is given, the last created directory is used
        if data_dir is None:
            data_dir = sorted(os.listdir('npy'))[-1]
        self.data_dir = data_dir
        if os.path.exists(f'{self.data_dir}/train') and os.path.exists(f'{self.data_dir}/test'):
            try:
                self.train_coarse_data = np.load(f'{self.data_dir}/train/CoarseResPoints.npy')
                self.train_high_data = np.load(f'{self.data_dir}/train/HighResPoints.npy')
                self.train_coarse_velocities = np.load(f'{self.data_dir}/train/CoarseResVelocities.npy')
                self.train_high_velocities = np.load(f'{self.data_dir}/train/HighResVelocities.npy')
                self.test_coarse_data = np.load(f'{self.data_dir}/test/CoarseResPoints.npy')
                self.test_high_data = np.load(f'{self.data_dir}/test/HighResPoints.npy')
                self.test_coarse_velocities = np.load(f'{self.data_dir}/test/CoarseResVelocities.npy')
                self.test_high_velocities = np.load(f'{self.data_dir}/test/HighResVelocities.npy')
            except FileNotFoundError:
                self.train_coarse_data = np.load(f'{self.data_dir}/train/CoarseResPoints_normalized.npy')
                self.train_high_data = np.load(f'{self.data_dir}/train/HighResPoints_normalized.npy')
                self.train_coarse_velocities = np.load(f'{self.data_dir}/train/CoarseResVelocities_normalized.npy')
                self.train_high_velocities = np.load(f'{self.data_dir}/train/HighResVelocities_normalized.npy')
                self.test_coarse_data = np.load(f'{self.data_dir}/test/CoarseResPoints_normalized.npy')
                self.test_high_data = np.load(f'{self.data_dir}/test/HighResPoints_normalized.npy')
                self.test_coarse_velocities = np.load(f'{self.data_dir}/test/CoarseResVelocities_normalized.npy')
                self.test_high_velocities = np.load(f'{self.data_dir}/test/HighResVelocities_normalized.npy')


    def sort_data(self):
        # Sorts the data in the data directory in order to mantain the same order between the low and high resolution datan with the structure CoarseResPoints_{norm}.npy and HighResPoints_{norm}.npy
        # The sorting is done by sorting only the norm of the external force
        if os.path.exists(f'{self.data_dir}/train') and os.path.exists(f'{self.data_dir}/test'):
            try:
                os.rmdir(f'{self.data_dir}/train')
                os.rmdir(f'{self.data_dir}/test')
            except OSError as e:
                shutil.rmtree(f'{self.data_dir}/train')
                shutil.rmtree(f'{self.data_dir}/test')
        files = os.listdir(self.data_dir)
        files.sort()     
        return files
    
    def compute_dimensions(self):
        files = self.sort_data()
        num_samples = len(files)//2
        coarse_data = np.load(f'{self.data_dir}/{files[0]}')
        high_data = np.load(f'{self.data_dir}/{files[num_samples]}')
        return coarse_data.shape, high_data.shape
    
    
    def join_data(self, train_size=0.8, normalize=False, efficient=False):
        # Joins the data from the data directory in order to have a single array with the data from both the low and high resolution
        norm = ''
        files = self.sort_data()
        sizes = self.compute_dimensions()
        num_samples = len(files)//4
        print(f'Number of samples: {num_samples}')
        print(f'Coarse data shape: {sizes[0]}')
        print(f'High data shape: {sizes[1]}')
        self.coarse_data = np.empty((num_samples, sizes[0][0], sizes[0][1]))
        self.high_data = np.empty((num_samples, sizes[1][0], sizes[1][1]))
        self.coarse_velocities = np.empty((num_samples, sizes[0][0], sizes[0][1]))
        self.high_velocities = np.empty((num_samples, sizes[1][0], sizes[1][1]))
        for i in range(0, num_samples):
            self.coarse_data[i] = np.load(f'{self.data_dir}/{files[i]}')  
            self.coarse_velocities[i] = np.load(f'{self.data_dir}/{files[i+num_samples]}')
            self.high_data[i] = np.load(f'{self.data_dir}/{files[i+2*num_samples]}')
            self.high_velocities[i] = np.load(f'{self.data_dir}/{files[i+3*num_samples]}')
            #os.remove(f'{self.data_dir}/{file}')
            #print(f"Joined data for {files[i]} and {files[i+num_samples]}")
        # Min max normalization
        if normalize:
            self.coarse_data = (self.coarse_data - np.min(self.coarse_data))/(np.max(self.coarse_data) - np.min(self.coarse_data))
            self.high_data = (self.high_data - np.min(self.high_data))/(np.max(self.high_data) - np.min(self.high_data))
            norm = '_normalized'
        # shuffle the data
        idx = np.random.permutation(num_samples)
        self.coarse_data = self.coarse_data[idx]
        self.high_data = self.high_data[idx]
        self.high_velocities = self.high_velocities[idx]
        self.coarse_velocities = self.coarse_velocities[idx]

        if not efficient:
            
            train_samples = int(num_samples*train_size)
            train_coarse = self.coarse_data[:train_samples]
            train_high = self.high_data[:train_samples]
            train_coarse_velocities = self.coarse_velocities[:train_samples]
            train_high_velocities = self.high_velocities[:train_samples]
            test_coarse = self.coarse_data[train_samples:]
            test_high = self.high_data[train_samples:]
            test_coarse_velocities = self.coarse_velocities[train_samples:]
            test_high_velocities = self.high_velocities[train_samples:]
            if not os.path.exists(f'{self.data_dir}/train'):
                os.mkdir(f'{self.data_dir}/train')
            if not os.path.exists(f'{self.data_dir}/test'):
                os.mkdir(f'{self.data_dir}/test')
            if normalize:
                np.save(f'{self.data_dir}/train/CoarseResPoints{norm}.npy', train_coarse)
                np.save(f'{self.data_dir}/train/HighResPoints{norm}.npy', train_high)
                np.save(f'{self.data_dir}/train/CoarseResVelocities{norm}.npy', train_coarse_velocities)
                np.save(f'{self.data_dir}/train/HighResVelocities{norm}.npy', train_high_velocities)
                np.save(f'{self.data_dir}/test/CoarseResPoints{norm}.npy', test_coarse)
                np.save(f'{self.data_dir}/test/HighResPoints{norm}.npy', test_high)
                np.save(f'{self.data_dir}/test/CoarseResVelocities{norm}.npy', test_coarse_velocities)
                np.save(f'{self.data_dir}/test/HighResVelocities{norm}.npy', test_high_velocities)
            else:
                np.save(f'{self.data_dir}/train/CoarseResPoints.npy', train_coarse)
                np.save(f'{self.data_dir}/train/HighResPoints.npy', train_high)
                np.save(f'{self.data_dir}/train/CoarseResVelocities.npy', train_coarse_velocities)
                np.save(f'{self.data_dir}/train/HighResVelocities.npy', train_high_velocities)
                np.save(f'{self.data_dir}/test/CoarseResPoints.npy', test_coarse)
                np.save(f'{self.data_dir}/test/HighResPoints.npy', test_high)
                np.save(f'{self.data_dir}/test/CoarseResVelocities.npy', test_coarse_velocities)
                np.save(f'{self.data_dir}/test/HighResVelocities.npy', test_high_velocities)
        else:
            self.train_coarse_data = self.coarse_data
            self.train_high_data = self.high_data
            self.train_coarse_velocities = self.coarse_velocities
            self.train_high_velocities = self.high_velocities
            if not os.path.exists(f'{self.data_dir}/train'):
                os.mkdir(f'{self.data_dir}/train')
            if normalize:
                np.save(f'{self.data_dir}/train/CoarseResPoints{norm}.npy', self.coarse_data)
                np.save(f'{self.data_dir}/train/HighResPoints{norm}.npy', self.high_data)
                np.save(f'{self.data_dir}/train/CoarseResVelocities{norm}.npy', self.coarse_velocities)
                np.save(f'{self.data_dir}/train/HighResVelocities{norm}.npy', self.high_velocities)
            else:
                np.save(f'{self.data_dir}/train/CoarseResPoints.npy', self.coarse_data)
                np.save(f'{self.data_dir}/train/HighResPoints.npy', self.high_data)
                np.save(f'{self.data_dir}/train/CoarseResVelocities.npy', self.coarse_velocities)
                np.save(f'{self.data_dir}/train/HighResVelocities.npy', self.high_velocities)
        

    


if __name__ == "__main__":
    # Example of usage
    dm = DataManager('npy/2024-07-30_09:12:39_dynamic_simulation')
    dm.join_data(normalize=False, efficient=True)
    dm2 = DataManager('npy/2024-07-30_09:12:39_dynamic_simulation')
    # reshape the data by flattenig the last two dimensions
    coarse_data = dm2.train_coarse_data.reshape(dm2.train_coarse_data.shape[0], -1)
    high_data = dm2.train_high_data.reshape(dm2.train_high_data.shape[0], -1)
    coarse_velocities = dm2.train_coarse_velocities.reshape(dm2.train_coarse_velocities.shape[0], -1)
    high_velocities = dm2.train_high_velocities.reshape(dm2.train_high_velocities.shape[0], -1)
    print(f'Coarse data shape: {coarse_data.shape}')
    print(f'High data shape: {high_data.shape}')
