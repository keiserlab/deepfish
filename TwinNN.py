import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    
    def __init__(self, margin_pos, margin_neg):
        super(ContrastiveLoss, self).__init__()
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        
    def forward(self, output1, output2, label):
        
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin_neg - euclidean_distance, min=0.0), 2))

        return (loss_contrastive,euclidean_distance)

class TwinNN(nn.Module):
    def __init__(self,feat_length):
        super(TwinNN, self).__init__()
        
        self.feat_length = feat_length
        
        self.fc2 = nn.Sequential(
            
            nn.Linear(self.feat_length, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(inplace=True),
            nn.Linear(4000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(inplace=True),
            nn.Linear(250, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10))
        
    def forward_once(self, x):

        output = self.fc2(x)
        #output = self.fc2(x)
        return output
        
    def forward(self, input1, input2):
        #print(input1.shape)
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

