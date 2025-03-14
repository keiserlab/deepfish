import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import os,sys,yaml
import numpy as np
import pandas as pd
from argparse import Namespace
from sklearn.metrics import roc_auc_score
from TwinNN import ContrastiveLoss,TwinNN
from TwinDN import _DenseLayer,_DenseBlock,_Transition,Flatten,PrintOutput,TwinDN
from TwinDataset import PairsDataset

with open('config.yaml') as cf_file:
    config = yaml.safe_load(cf_file.read())
config = Namespace(**config)

num_gpus = int(config.num_gpus) #Select number of GPUs
gpu_ids = str(config.gpu_ids)
batch_size = int(config.batch_size) #Batch size
batch_size = int(batch_size * num_gpus) 
dense_flag = bool(config.dense_flag) #Use Twin-NN or Twin-DN model (0 for Twin-DN)
learning_rate = float(config.learning_rate)
weight_decay = float(config.weight_decay)
epochs = int(config.epochs)
log_interval = int(config.log_interval)
save_flag = bool(config.save_flag) #Save model to disk?
save_epoch_interval = int(config.save_epoch_interval) #epoch frequency for saving model
data_dir = str(config.data_dir)
results_dir = str(config.results_dir)
save_tag = str(config.save_tag)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
torch.cuda.set_device(0)

if dense_flag == 0:
    from TwinNN import *
else:
    from TwinDN import *

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

model_path = results_dir + save_tag

if not os.path.exists(model_path):
    os.mkdir(model_path)

df_info= pd.read_csv('{}NT650_info.csv'.format(data_dir))
cols = ['run','well_index']
full_info = [tuple(x) for x in df_info[cols].to_records(index=False)]
full_info = np.array(full_info)
full_info[:,1] = full_info[:,1] -1 

pairs_file_train = '{}pairs_train.npy'.format(data_dir + 'pairs_filtered' + '/')
labels_file_train = '{}labels_train.npy'.format(data_dir + 'pairs_filtered' + '/')

pairs_file_test = '{}pairs_test.npy'.format(data_dir + 'pairs_filtered' + '/')
labels_file_test = '{}labels_test.npy'.format(data_dir + 'pairs_filtered' + '/')

data_file = '{}NT650_MI_data.npy'.format(data_dir)
fp_arr = np.load(data_file)[:,::5] #Take every fifth frame from full time-series

min_ = np.min(fp_arr)
max_ = np.max(fp_arr)
fp_arr = (fp_arr - min_) / (max_ - min_)
feat_length = fp_arr.shape[1]
        
print(fp_arr.shape, feat_length)

train_dataset = PairsDataset(pairs_file=pairs_file_train,
                                 labels_file=labels_file_train,
                                 data=fp_arr,
                                 dense_flag=dense_flag,
                                 full_info = full_info)
test_dataset = PairsDataset(pairs_file=pairs_file_test,
                                 labels_file=labels_file_test,
                                 data=fp_arr,
                                 dense_flag=dense_flag,
                                 full_info = full_info)

labels_train_ = np.load(labels_file_train).reshape(-1).tolist()
labels_test_ = np.load(labels_file_test).reshape(-1).tolist()

kwargs = {'num_workers': 16, 'pin_memory': True}
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, drop_last=False, **kwargs)

print(len(train_loader),len(test_loader))
print('feat_length',feat_length)

if dense_flag == 0:
    net = TwinNN(feat_length).cuda()
    print('Using Twin-NN')
else:
    net = TwinDN().cuda()
    print('Using Twin-DN')

net = nn.DataParallel(net,device_ids=list(range(num_gpus)))
criterion = ContrastiveLoss(margin_pos = 0, margin_neg = 0.5)
optimizer = optim.Adam(net.parameters(),lr=learning_rate,amsgrad=False,weight_decay=weight_decay)

def get_n_params(net):
    pp=0
    for p in list(net.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
print(get_n_params(net))

def train(net):
    net.train()
    train_loss = 0
    for i, data_train in enumerate(train_loader,0):
        
        if i % log_interval == 0:
            print('{} out of {}'.format(i*batch_size,len(train_loader)*batch_size))
        data0 = data_train[0][0][:,:].float()
        data1 = data_train[0][1][:,:].float()
        label = data_train[1].float()
        data0, data1 , label = Variable(data0).cuda(), Variable(data1).cuda() , Variable(label).cuda() 
        label = label.view(-1)
        optimizer.zero_grad()
        output1,output2 = net(data0,data1)
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive[0].backward()
        optimizer.step()
        train_loss += loss_contrastive[0].data
        
    train_loss /= len(train_loader)
    print("Epoch number {}\n Train loss {}\n".format(epoch,train_loss))
    return train_loss

def test(net):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data_test in enumerate(test_loader,0):

            data0 = data_test[0][0][:,:].float()
            data1 = data_test[0][1][:,:].float()
            label = data_test[1].float()
            data0, data1 , label = Variable(data0).cuda(), Variable(data1).cuda() , Variable(label).cuda()
            label = label.view(-1)
            output1,output2 = net(data0,data1)
            loss_contrastive = criterion(output1,output2,label)
            test_loss += loss_contrastive[0].data
            
    test_loss/= len(test_loader)
            
    #print("Epoch number {}\n Test loss {}\n".format(epoch,test_loss))
    return test_loss

log_file = open(model_path + 'training.log','w')
log_file.write('epoch,train_loss,val_loss\n')

counter = []
iteration_number= 0

def calc_margin(net):
    net.eval()
    test_loss = 0
    dists_p = []
    dists_n = []
    dists_all = []
    labels_all = []
    with torch.no_grad():
        for i, data_test in enumerate(test_loader,0):
            data0 = data_test[0][0][:,:].float()
            data1 = data_test[0][1][:,:].float()
            label = data_test[1].float()
            data0, data1 , label = Variable(data0).cuda(), Variable(data1).cuda() , Variable(label).cuda()
            label = label.view(-1)
            output1,output2 = net(data0,data1)
            loss_contrastive = criterion(output1,output2,label)
            labels = label.data.cpu().numpy()
            dists = loss_contrastive[1].data.cpu().numpy()
            for il,l in enumerate(labels):
                labels_all.append(l)
                dists_all.append(dists[il])
                if int(l) == 0:
                    dists_p.append(dists[il])
                else:
                    dists_n.append(dists[il])
    return dists_p, dists_n, dists_all, labels_all

for epoch in range(1,epochs+1):
    
    train_loss = train(net)
    test_loss = test(net)
    
    dists_p, dists_n, dists, labels_test = calc_margin(net)
    roc_score = roc_auc_score(labels_test, dists)
    
    if (epoch) % save_epoch_interval == 0 and (epoch) >= 1 and save_flag == 1:
        torch.save(net.state_dict(), model_path + 'state_dict_epoch_{}.pt'.format(epoch))
    
    print(epoch,train_loss.data.cpu().numpy(),test_loss.data.cpu().numpy(),roc_score)
    
    log_file = open(model_path + 'training.log','a')
    log_file.write('{},{:.4f},{:.4f}\n'.format(epoch,train_loss,test_loss))
    
    
