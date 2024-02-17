from torch.utils.data import DataLoader,Dataset
import numpy as np

class PairsDataset(Dataset):

    def __init__(self, pairs_file, labels_file, data, dense_flag, full_info):
        
        self.pairs_file = np.load(pairs_file)
        self.labels_file = np.load(labels_file)
        self.data_file = data
        self.dense_flag = dense_flag
        self.full_info = full_info

    def __len__(self):
        return len(self.pairs_file)
    
    def __getitem__(self, idx):
        
        index1 = int(self.pairs_file[idx][0]); index2 = int(self.pairs_file[idx][1])
        label = self.labels_file[idx].astype('float')
        tr1 = self.data_file[index1].astype('float'); tr2 = self.data_file[index2].astype('float')
        pair = (tr1,tr2)
        rand = np.random.randint(0,2,1)[0]
        pair1 = pair[rand]; pair2 = pair[1-rand]
        pair_new = (pair1,pair2)
        tr1 = pair_new[0]; tr2 = pair_new[1]
        
        if self.dense_flag:
            tr1 = tr1.reshape(1,-1); tr2 = tr2.reshape(1,-1)
        else:
            tr1 = tr1.reshape(-1); tr2 = tr2.reshape(-1)
        return (tr1,tr2),label