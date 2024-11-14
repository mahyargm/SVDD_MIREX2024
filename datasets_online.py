import numpy as np
import os
from torch.utils.data import Dataset
import random
import soundfile as sf
from torch import Tensor
import traceback
import librosa

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

class SVDD2024(Dataset):
    """
    Dataset class for the SVDD 2024 dataset.
    """
    def __init__(self, base_dir, partition="mixture", max_len=44100*4, test_flag = False):
        assert partition in ["vocals", "mixture"], "Invalid partition. Must be one of ['vocals', 'mixture']"
        self.base_dir = base_dir
        self.partition = partition
        self.base_dir = os.path.join(base_dir, partition)
        self.balance_batch = True
        self.test_flag = test_flag
        self.max_len = max_len
        self.file_list = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                self.file_list.append(os.path.join(root, file))
       
        self.deepfakes = [item for item in self.file_list if item.split("/")[-1][0]=='d']
        self.bonafides = [item for item in self.file_list if item.split("/")[-1][0]=='b']
        self.bonafides_length = len(self.bonafides)
        self.deepfakes_length = len(self.deepfakes)



    def __len__(self):
        if self.test_flag:
            return len(self.file_list)
        else:
            return max(self.deepfakes_length, self.bonafides_length) * 2


    def __getitem__(self, index):
        if self.test_flag:
            file_path = self.file_list[index].strip()
            if file_path.split("/")[-1][0]=='d':
                label = 0
            else:
                label = 1
        else:
            if index%2 == 0: # Bonafide
                if index/2 < self.bonafides_length:
                    file = self.bonafides[index//2]
                else: file = random.choice(self.bonafides)
                file_path = file.strip()
                label = 1
            else:            # DeepFake
                if (index-1)/2 < self.deepfakes_length:
                    file = self.deepfakes[(index-1)//2]
                else: file = random.choice(self.deepfakes)
                file_path = file.strip()
                label = 0


        try:                    # SoundFile
            audio_len = len(sf.SoundFile(file_path))
            if audio_len<=self.max_len:
                x, sr = sf.read(file_path)
                if len(x.shape) == 2:
                    x = x.sum(axis=1) / 2
                x = pad(x, self.max_len)
            else: 
                start_sample = random.randrange(start=0, stop=audio_len - self.max_len)
                x, sr = sf.read(file_path, start = start_sample, stop = start_sample + self.max_len)
                if len(x.shape) == 2:
                    x = x.sum(axis=1) / 2

        except Exception as e:   # Librosa
            print(file_path)
            x, sr = librosa.load(file_path)
            audio_len = len(x)
            if len(x.shape) == 2:
                x = x.sum(axis=1) / 2
            if audio_len<=self.max_len:
                x = pad(x, self.max_len)
            else:
                x = x[start_sample:start_sample + (self.max_len)]
            
            print(f"Error in loading {file_path} has been handled: {e}")
            
                

        if sr!=44100: print('Error! Sample Rate should be 44100!')
        x = Tensor(x)
        return x, label, file_path.split('/')[-1].strip()


