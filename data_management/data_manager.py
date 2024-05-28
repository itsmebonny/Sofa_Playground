import numpy as np
import os
import shutil






class DataManager:

    def __init__(self, data_dir = None) -> None:
        # Creates a DataManager object that takes as input the directory where the data is stored
        # If no directory is given, the last created directory is used
        # check if data_dir contains "efficient" in the name
        if data_dir is None:
            data_dir = sorted(os.listdir('npy'))[-1]
        self.data_dir = data_dir

        if self.data_dir.find('efficient') != -1:
            self.efficient = True
        else:
            self.efficient = False

        if os.path.exists(f'{self.data_dir}/train') and os.path.exists(f'{self.data_dir}/test'):
            try:
                self.train_coarse_data = np.load(f'{self.data_dir}/train/CoarseResPoints.npy')
                self.train_high_data = np.load(f'{self.data_dir}/train/HighResPoints.npy')
                self.test_coarse_data = np.load(f'{self.data_dir}/test/CoarseResPoints.npy')
                self.test_high_data = np.load(f'{self.data_dir}/test/HighResPoints.npy')
            except FileNotFoundError:
                self.train_coarse_data = np.load(f'{self.data_dir}/train/CoarseResPoints_normalized.npy')
                self.train_high_data = np.load(f'{self.data_dir}/train/HighResPoints_normalized.npy')
                self.test_coarse_data = np.load(f'{self.data_dir}/test/CoarseResPoints_normalized.npy')
                self.test_high_data = np.load(f'{self.data_dir}/test/HighResPoints_normalized.npy')
        elif os.path.exists(f'{self.data_dir}/train'):
            try:
                self.train_coarse_data = np.load(f'{self.data_dir}/train/CoarseResPoints.npy')
                self.train_high_data = np.load(f'{self.data_dir}/train/HighResPoints.npy')
            except FileNotFoundError:
                self.train_coarse_data = np.load(f'{self.data_dir}/train/CoarseResPoints_normalized.npy')
                self.train_high_data = np.load(f'{self.data_dir}/train/HighResPoints_normalized.npy')
        else:
            self.train_coarse_data = None
            self.train_high_data = None
            self.test_coarse_data = None
            self.test_high_data = None
            print("No data found")


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
        self.coarse_data = np.empty((len(files)//2, sizes[0][0], sizes[0][1]))
        self.high_data = np.empty((len(files)//2, sizes[1][0], sizes[1][1]))
        num_samples = len(files)//2
        for i in range(0, num_samples):
            self.coarse_data[i] = np.load(f'{self.data_dir}/{files[i]}')  
            self.high_data[i] = np.load(f'{self.data_dir}/{files[i+num_samples]}')
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
        if efficient:
            print("Efficient mode")
            self.train_coarse_data = self.coarse_data
            self.train_high_data = self.high_data
            if not os.path.exists(f'{self.data_dir}/train'):
                os.mkdir(f'{self.data_dir}/train')
            np.save(f'{self.data_dir}/train/CoarseResPoints.npy', self.train_coarse_data)
            np.save(f'{self.data_dir}/train/HighResPoints.npy', self.train_high_data)
        else:
            train_samples = int(num_samples*train_size)
            train_coarse = self.coarse_data[:train_samples]
            train_high = self.high_data[:train_samples]
            test_coarse = self.coarse_data[train_samples:]
            test_high = self.high_data[train_samples:]
            if not os.path.exists(f'{self.data_dir}/train'):
                os.mkdir(f'{self.data_dir}/train')
            if not os.path.exists(f'{self.data_dir}/test'):
                os.mkdir(f'{self.data_dir}/test')
            if normalize:
                np.save(f'{self.data_dir}/train/CoarseResPoints{norm}.npy', train_coarse)
                np.save(f'{self.data_dir}/train/HighResPoints{norm}.npy', train_high)
                np.save(f'{self.data_dir}/test/CoarseResPoints{norm}.npy', test_coarse)
                np.save(f'{self.data_dir}/test/HighResPoints{norm}.npy', test_high)
            else:
                np.save(f'{self.data_dir}/train/CoarseResPoints.npy', train_coarse)
                np.save(f'{self.data_dir}/train/HighResPoints.npy', train_high)
                np.save(f'{self.data_dir}/test/CoarseResPoints.npy', test_coarse)
                np.save(f'{self.data_dir}/test/HighResPoints.npy', test_high)

        

    


if __name__ == "__main__":
    # Example of usage
    dm = DataManager('npy_gmsh/2024-05-28_11:10:16_estimation_efficient_183nodes')
    dm.join_data(efficient=dm.efficient)
    dm2 = DataManager('npy_gmsh/2024-05-28_11:10:16_estimation_efficient_183nodes')
    # reshape the data by flattenig the last two dimensions
    coarse_data = dm2.train_coarse_data.reshape(dm2.train_coarse_data.shape[0], -1)
    high_data = dm2.train_high_data.reshape(dm2.train_high_data.shape[0], -1)
    print(f'Coarse data shape: {coarse_data.shape}')
    print(f'High data shape: {high_data.shape}')
