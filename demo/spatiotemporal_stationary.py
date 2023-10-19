import torch
import tqdm
import gpytorch
import urllib.request
import os
from math import floor
import pandas as pd
import numpy as np
import models.dgps as m
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls import DeepApproximateMLL
from gpytorch.kernels import RBFKernel, ScaleKernel, PeriodicKernel, MaternKernel
from sklearn.utils import shuffle
import scipy.stats
import cartopy.crs as ccrs
from scipy.special import inv_boxcox
from utils.plotting import plot_spatio_temporal_predictions
from utils.metrics import nlpd, rmse, negative_log_predictive_density
from gpytorch.constraints import GreaterThan
from utils.config import RESULTS_DIR, DATASET_DIR

gpytorch.settings.cholesky_max_tries(4)

filepath = 'data/uib_spatio_temporal.csv'

# Kernel
kernel = gpytorch.kernels.ScaleKernel(RBFKernel(ard_num_dims=2, active_dims=[1,2])+ gpytorch.kernels.RBFKernel(active_dims=[1,2]) * gpytorch.kernels.PeriodicKernel(active_dims=[0]))

def load_uib_spatio_temporal_data():
        
    fname = str(DATASET_DIR) + '/uib_spatio_temporal.csv'
    data = pd.read_csv(fname)
    return data, torch.Tensor(np.array(data))[:,0:3], torch.Tensor(np.array(data)[:,-1])

def load_uib_train_test():
        
    data, x, y = load_uib_spatio_temporal_data()
    data = data[data['time'] < 2001]
    data['month'] = data['time'].rank(method='dense').astype('int')
    train_test_data = data[data['month'] < 6]
    x, y = torch.Tensor(np.array(train_test_data)).double()[:,0:3], torch.Tensor(np.array(train_test_data)[:,-2]).double()
    
    with torch.no_grad():
        stdx, meanx = torch.std_mean(x, dim=-2)
        x_norm = (x -  meanx) / stdx
        stdy, meany = torch.std_mean(y)
        y_norm = (y - meany) / stdy
        
    split_idx = len(np.where(train_test_data['month'] < 5)[0])
    x_train, y_train = x_norm[0:split_idx], y_norm[0:split_idx]
    x_test, y_test = x_norm[split_idx:], y_norm[split_idx:]
    return train_test_data, x_train, y_train, x_test, y_test, meany, stdy, x_norm, y

if __name__ == "__main__":

    data, train_x, train_y, test_x, test_y, meany, stdy, x_norm, y = load_uib_train_test()

    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y, x_norm = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda(), x_norm.cuda()
    
    
    #### Model
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    model = m.ExactGPModel(train_x, train_y, likelihood, kernel).double()
    #model.mean_module.register_constraint('constant', gpytorch.constraints.GreaterThan(0.5))
    #model.mean_module.initialize(constant=0.5)
    #### Training
    training_iter = 1200
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    # Parameters to optimise
    training_parameters =  model.parameters() # all parameters
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(training_parameters , lr=0.03)  # Includes GaussianLikelihood parameters
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)   
    
    losses = []
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        losses.append(loss.item())
        if i%100 == 0:
            print('Iter %d/%d - Loss: %.3f'  # lengthscale: %.3f   noise: %.3f' 
                % (i + 1, training_iter, loss.item()))
                #model.covar_module.base_kernel.lengthscale.item(),
                #model.likelihood.noise.item()))
        optimizer.step()
     
    #### Metrics
    
    model.eval()
    with torch.no_grad():
        pred_y_test = likelihood(model(test_x)) 
        pred_y_full = likelihood(model(x_norm))
        y_mean = pred_y_test.loc.detach()
        y_mean_full = pred_y_full.loc.detach()
        y_std = pred_y_test.covariance_matrix.diag().sqrt().detach()
    
    # Inverse transform predictions
    # # pred_y_test_tr = torch.Tensor(inv_boxcox(pred_y_test, bc_param))
    # #y_mean_tr = torch.Tensor(inv_boxcox(y_mean, bc_param))
    # #y_var_tr = torch.Tensor(inv_boxcox(y_var + y_mean, bc_param,)) - y_mean_tr
    # #test_y_tr = torch.Tensor(inv_boxcox(test_y, bc_param))
  
    # ## Metrics
    rmse_test = rmse(y_mean, test_y, stdy)
        
    nlpd_norm = negative_log_predictive_density(test_y, y_mean, y_std**2, stdy)
    nlpd_og = negative_log_predictive_density(test_y*stdy + meany, y_mean*stdy + meany, (y_std*stdy)**2, torch.Tensor([1.0]))
  
    assert((nlpd_norm - nlpd_og) < 1e-5)
    
    ### Viz
    final_mean = y_mean_full*stdy + meany
    plot_spatio_temporal_predictions(data, 'Stationary Kernel: ' + r'$k\_se(lat, lon) + k\_se(lat, lon)*k\_per(time)$', final_mean)
    plot_spatio_temporal_predictions(data, 'Ground Truth', data['tp'])