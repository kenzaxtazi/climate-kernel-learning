#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UIB Spatio-Temporal model

"""

import numpy as np
import math
import torch 
import gpytorch
import pandas as pd
import getopt
import sys
import matplotlib.pylab as plt
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel, InducingPointKernel
from models.gibbs_kernels import LogNormalPriorProcess, PositivePriorProcess, GibbsKernel, GibbsSafeScaleKernel
from models.spatio_temporal_models import SpatioTemporal_Stationary, SparseSpatioTemporal_Nonstationary
from gpytorch.constraints import GreaterThan

from utils.config import BASE_SEED, EPSILON, DATASET_DIR, RESULTS_DIR
from utils.metrics import rmse, nlpd, negative_log_predictive_density,get_trainable_param_names
from utils.plotting import plot_spatio_temporal_predictions

gpytorch.settings.cholesky_jitter(EPSILON)

## helper methods and classes 

def load_uib_spatio_temporal_data():
        
    fname = str(DATASET_DIR) + '/uib_spatio_temporal.csv'
    data = pd.read_csv(fname)
    return data, torch.Tensor(np.array(data))[:,0:3], torch.Tensor(np.array(data)[:,-1])

def load_uib_train_test():
        
    data, x, y = load_uib_spatio_temporal_data()
    data = data[data['time'] < 2001]
    data['month'] = data['time'].rank(method='dense').astype('int')
    train_test_data = data[data['month'] < 6]
    x, y = torch.Tensor(np.array(train_test_data))[:,0:3], torch.Tensor(np.array(train_test_data)[:,-2])
    
    with torch.no_grad():
        stdx, meanx = torch.std_mean(x, dim=-2)
        x_norm = (x -  meanx) / stdx
        stdy, meany = torch.std_mean(y)
        y_norm = (y - meany) / stdy
        
    #split_idx = len(np.where(train_test_data['time'] < test_year)[0])
    #test_idx = np.where(train_test_data['time'].astype('int') == test_year)[0]
    split_idx = len(np.where(train_test_data['month'] < 5)[0])
    x_train, y_train = x_norm[0:split_idx], y_norm[0:split_idx]
    x_test, y_test = x_norm[split_idx:], y_norm[split_idx:]
    return train_test_data, x_train, y_train, x_test, y_test, meany, stdy, x_norm, y
    
def parse_args(argv):
    
    """Get command line arguments, set defaults and return a dict of settings."""

    args_dict = {
        ## File paths and logging
        
            'data'          : '/data/',            # relative path to data dir from root
            'device'        : 'cpu',               # cpu or cuda
        
        ## Training options
        
            'model'         : 'Non-Stationary',        # 'Stationary' / 'Non-Stationary'
            'lr'            : '1e-2',              # learning rate          
            'max_iters'     : 1000,
            'threshold'     : 1e-6,                # improvement after which to stop
            'M'             : 500,                # Number of inducing points (sparse regression only)
            'prior_scale'   : 1,                   # initial value for the prior outputscale (same for both dims)
            'prior_ell'     : 1.3,                 # Initial value for the prior's lengthscale (same for both dims)
            'prior_mean'    : 0.3,                 # Initial value for the prior's mean (same for both dims)
            'noise'         : 0.0,                 # 0 for optimised noise, else fixed through training
            'scale'         : 0 ,                  # 0 for optimised output scale, else fixed through training
    }

    try:
        opts, _ = getopt.getopt(argv, '', [name + '=' for name in args_dict.keys()])
    except getopt.GetoptError:
        helpstr = 'Check options. Permitted long options are '
        print(helpstr, args_dict.keys())

    for opt, arg in opts:
        opt_name = opt[2:] # remove initial --
        args_dict[opt_name] = arg
    
    return args_dict
        
if __name__ == "__main__":
    
    args = parse_args(sys.argv[1:])
    device = args['device']
    max_cg_iterations = 4000
    
    #### Loading and prep data
    
    data, x_train, y_train, x_test, y_test, meany, stdy, x_norm, y = load_uib_train_test()
    
    #num_inducing = 500  
    z = x_train[::2]
    #z = torch.tensor(pm.gp.util.kmeans_inducing_points(num_inducing, np.array(x_train)))
    #z = None
    ## Initialising model-set up and prior settings
    
    if args['model'] == 'Non-Stationary':
        
        prior = LogNormalPriorProcess(input_dim=2).to(device)
            #### change the prior settings here if desired
        prior.covar_module.outputscale = float(args['prior_scale']) * torch.ones_like(prior.covar_module.outputscale)
        prior.covar_module.base_kernel.lengthscale = float(args['prior_ell']) * torch.ones_like(
                                                                prior.covar_module.base_kernel.lengthscale)
        prior.mean_module.constant = torch.nn.Parameter(
            math.log(float(args['prior_mean'])) * torch.ones_like(prior.mean_module.constant)
                                )
     
        for p in prior.parameters():
            p.requires_grad = False
        
    ## Training 
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if args['model'] == 'Stationary':
        model = SpatioTemporal_Stationary(x_train, y_train, likelihood, z) 
    else:
        model = SparseSpatioTemporal_Nonstationary(x_train, y_train, likelihood, prior, z, num_dim=2) 
        
    #model.spatial_covar_module(x_train[:,(1,2)], ell=torch.exp(model.log_ell_z)).evaluate()
    #odel.temporal_covar_module(x_train).evaluate()
    

    # #noise hyper
    # if float(args['noise']) > 0:
    #     model.likelihood.noise = float(args['noise'])
    #     for p in model.likelihood.noise_covar.parameters():
    #         p.requires_grad = False
    # # outputscale hyper
    # if float(args['scale']) > 0:
    #     model.covar_module.outputscale = float(args['scale'])
    #     model.covar_module._parameters['raw_outputscale'].requires_grad = False
           
    model.train()
    likelihood.train()
    
    n_iter = 500
     
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    losses = []
    for i in range(n_iter):
        optimizer.zero_grad()
        with gpytorch.settings.max_cg_iterations(max_cg_iterations):
            output = model(x_train)
            loss = -mll(output, y_train)
        loss.backward()
        losses.append(loss.item())
        if i%50 == 0:
            print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
                i + 1, n_iter, loss.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()
    
     ## Testing
     
    model.eval()
    likelihood.eval()
    
    if args['model'] == 'Non-Stationary':
        pred_y_test = likelihood(model.predict(x_test)) 
    else:
        pred_y_test = likelihood(model(x_test)) 
     
    y_mean = pred_y_test.loc.detach()
    y_var = pred_y_test.covariance_matrix.diag().detach()
    
    ## Metrics
    rmse_test = rmse(y_mean, y_test, stdy)
    nlpd_test = negative_log_predictive_density(y_test, y_mean, y_var, stdy)
    
    print('RMSE test =  ' + str(rmse_test))
    print('NLPD test = ' + str(nlpd_test))
    
    # ### Pred full
    
    if args['model'] == 'Non-stationary':
        pred_f = likelihood(model.predict(x_norm))
    else:
        pred_f = likelihood(model(x_norm))
    
    f_mean = pred_f.loc.detach()
    f_var = pred_f.covariance_matrix.diag().detach()
    final_mean = f_mean*stdy + meany
    np.savetxt('results/f_mean_ns.csv',final_mean)

    plot_spatio_temporal_predictions(title='Non-stationary Kernel', final_mean=f_mean*stdy + meany)
   
    