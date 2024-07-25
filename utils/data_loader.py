import numpy as np
import keras


class DataLoader(keras.utils.PyDataset):
    def __init__(self, x, y, batch_size, augment, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = x, y
        self.x = self.x[:,132:,:] - self.x[:,131:132,:]
        self.sub_x, self.sub_y = np.copy(x), np.copy(y)
        self.batch_size = batch_size
        self.augment = augment
        if self.augment:
            self.shuffle_x_rng = np.random.default_rng(0)
            self.shuffle_y_rng = np.random.default_rng(0)
            self.subset_rng = np.random.default_rng(42)
            self.slice_rng = np.random.default_rng(1729)
        else:
            mean = np.expand_dims(self.sub_x.mean(axis=1), axis=1)
            std = np.expand_dims(self.sub_x.std(axis=1), axis=1)
            self.sub_x = (self.sub_x - mean) / std

    def get_random_ranges(self, arr, K):
        shape = arr.shape
        start_indices = np.random.randint(0, shape[1] - K + 1, size=(shape[0],))
        selected_ranges = np.zeros((shape[0], K, shape[2]))
        for i in range(shape[0]):
            start_idx = start_indices[i]
            selected_ranges[i] = arr[i, start_idx : start_idx + K, :]
        return selected_ranges

    def on_epoch_end(self):
        if self.augment:
            self.sub_x, self.sub_y = np.copy(self.x), np.copy(self.y)
            self.shuffle_x_rng.shuffle(self.sub_x)
            self.shuffle_y_rng.shuffle(self.sub_y)
            choice = self.subset_rng.choice([0, 1, 2, 3, 4, 5])
            if choice in [0, 1, 2]:
                self.sub_x = self.sub_x[:, choice * 80 : (choice + 1) * 80, :]
            elif choice in [3, 4]:
                self.sub_x = self.sub_x[:, (choice - 3) * 80 : (choice - 1) * 80, :]
            total_len = self.sub_x.shape[1]
            K = self.slice_rng.integers(8 * total_len // 10, total_len + 1)
            self.sub_x = self.get_random_ranges(self.sub_x, K)
            mean = np.expand_dims(self.sub_x.mean(axis=1), axis=1)
            std = np.expand_dims(self.sub_x.std(axis=1), axis=1)
            self.sub_x = (self.sub_x - mean) / std

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = low + self.batch_size
        batch_x = self.sub_x[low:high]
        batch_y = self.sub_y[low:high]
        return batch_x, batch_y
