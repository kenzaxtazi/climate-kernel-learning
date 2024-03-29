#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computation of metrics

"""

import torch
from prettytable import PrettyTable

def print_trainable_param_names(model):

    ''' Prints a list of parameters (model + variational) which will be
    learnt in the process of optimising the objective '''

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")


def get_trainable_param_names(model):

    ''' Returns a list of names for the model which are learnable '''
    grad_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        grad_params.append(name)
    return grad_params

def rmse(Y_pred_mean, Y_test, Y_std):

      return Y_std.item()*torch.sqrt(torch.mean((Y_pred_mean - Y_test)**2)).detach()
  
def nlpd(Y_test_pred, Y_test, Y_std):
    
      lpd = Y_test_pred.log_prob(Y_test)
      # return the average
      avg_lpd_rescaled = lpd.detach()/len(Y_test) - torch.log(Y_std)
      return -avg_lpd_rescaled
  
    
def negative_log_predictive_density(test_y, predicted_mean, predicted_var , Y_std):
   # Vector of log-predictive density per test point    
   lpd = torch.distributions.Normal(predicted_mean, torch.sqrt(predicted_var)).log_prob(test_y) 
   # return the average
   avg_lpd_rescaled = torch.mean(lpd)- torch.log(Y_std)
   return -avg_lpd_rescaled

    