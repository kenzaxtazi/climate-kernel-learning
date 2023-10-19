# Data preparation

import pandas as pd
import scipy as sp
import numpy as np
import torch
import math
from torch.utils.data import TensorDataset
from utils.config import DATASET_DIR

def download_data(filepath):
    
    df = pd.read_csv(filepath)
    data = torch.Tensor(df.values)
    return data

def load_per_region_timeseries(region):
    
    fname = str(DATASET_DIR) + '/' + region + '_time_series.csv'
    data = pd.read_csv(fname)
    return torch.Tensor(np.array(data))[:,0], torch.Tensor(np.array(data)[:,-1])

def prep_inputs(data):
    
    # X = data[:, 1:-1]
    # X = X - X.min(0)[0]
    # X = 2 * (X / X.max(0)[0]) - 1
    x = data[:,:-1]
    stdx, meanx = torch.std_mean(x, dim=-2)
    x_norm = (x -  meanx) / stdx
    return x_norm

def prep_outputs(data):
    y = data[:, -1]
    # performs Box-Cox transformation to make y distribution more Gaussian
    y_tr, bc_param = sp.stats.boxcox(y)
    # y = sp.special.inv_boxcox(y_tr, bc_param)
    return y_tr, bc_param

def box_cox_transform(data):
    
    return prep_inputs(data), prep_outputs(data)

def whitening_transform(data):
    
    x = data[:,:-1]
    y = data[:,-1]
    stdx, meanx = torch.std_mean(x, dim=-2)
    x_norm = (x -  meanx) / stdx
    stdy, meany = torch.std_mean(y)
    y_norm = (y - meany) / stdy
    return x_norm, y_norm, meanx, stdx, meany, stdy 

def train_test_split(X, y, train_prop):
    
    train_n = int(math.floor(train_prop * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()
    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()
    return train_x, train_y, test_x, test_y 
    
    
if __name__ == "__main__":
    
    filepath = 'data/uib_spatial.csv'
    data = download_data(filepath)
    
    
#     train_x, train_y, test_x, test_y = test_train_split(X, y)
    
#     if torch.cuda.is_available():
#         train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
    
#     train_dataset = TensorDataset(train_x, train_y)
    