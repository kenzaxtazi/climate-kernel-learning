#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatio-Temporal Models

Stationary and Non-stationary case

"""
import math
import torch
import gpytorch
from models.gibbs_kernels import GibbsKernel, GibbsSafeScaleKernel, InducingGibbsKernelST, InducingGibbsKernel
from utils import functional as fn
from gpytorch.kernels import ScaleKernel, PeriodicKernel, RBFKernel, InducingPointKernel
from gpytorch.constraints import GreaterThan

class SpatioTemporal_Stationary(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, z=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.temporal_covar_module = ScaleKernel(RBFKernel(active_dims=(1,2))*PeriodicKernel(active_dims=(0)), outputscale_constraint=GreaterThan(7
            ), active_dims=0)
        self.spatial_covar_module = ScaleKernel(RBFKernel(active_dims=(1,2)), active_dims=(1,2))
        if z is not None:
            self.covar_module = InducingPointKernel(base_kernel = self.temporal_covar_module + self.spatial_covar_module, inducing_points = z, likelihood=likelihood)
        else:
            self.covar_module = self.temporal_covar_module + self.spatial_covar_module
        
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
class SparseSpatioTemporal_Nonstationary(gpytorch.models.ExactGP):
    """
        Model for MAP inference of sparse Gibbs kernel GP.
    
    """
    def __init__(self, train_x, train_y, likelihood, prior, z, num_dim=1):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.spatial_covar_module = GibbsSafeScaleKernel(InducingGibbsKernel(GibbsKernel(lengthscale_prior=prior), inducing_points=z[:,(1,2)], likelihood=likelihood))
        #self.temporal_covar_module = InducingPointKernel(ScaleKernel(RBFKernel(active_dims=(1,2))*PeriodicKernel(active_dims=(0)),
        #                                         outputscale_constraint=GreaterThan(7)), inducing_points=z, likelihood=likelihood)
        self.temporal_covar_module = ScaleKernel(RBFKernel(ard_num_dims=2,active_dims=(1,2))*PeriodicKernel(active_dims=(0)))
        #self.temporal_covar_module.inducing_points.requires_grad = False
        self.spatial_covar_module.base_kernel.inducing_points.requires_grad = False

        #self.covar_module = self.spatial_covar_module + self.temporal_covar_module
            
        self.register_parameter('log_ell_z', 
                                torch.nn.Parameter(
                                    self.spatial_covar_module.base_kernel.base_kernel.lengthscale_prior.forward(z[:,(1,2)]).mean.clone()
                                ))
        self.register_prior('ell_z_prior', 
                            self.spatial_covar_module.base_kernel.base_kernel.lengthscale_prior, 
                            lambda module : (module.spatial_covar_module.base_kernel.inducing_points, module.log_ell_z)
                           )
        
    def forward(self, x, ell=None):
        softplus = torch.nn.Softplus()
        mean = softplus(self.mean_module(x))
        covar = self.temporal_covar_module(x) + self.spatial_covar_module(x[:,(1,2)], ell=torch.exp(self.log_ell_z))
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
    def predict(self, x_new):
        """Returns predictive at x_new given estimate of lengthscales at inducing points.
        WARNING: currently only the marginals of this are correct"""
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [x_new.unsqueeze(-1) if x_new.ndimension() == 1 else x_new]
        
        # Concatenate the input to the training input
        #active_dims = self.covar_module.kernels[0].active_dims
        active_dims = (1,2)
        full_inputs = []
        ell_cond = []
        batch_shape = train_inputs[0].shape[:-2]
        for train_input, input in zip(train_inputs, inputs):
            # Make sure the batch shapes agree for training/test data
            if batch_shape != train_input.shape[:-2]:
                batch_shape = gpytorch.utils.broadcasting._mul_broadcast_shape(batch_shape, train_input.shape[:-2])
                train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
            if batch_shape != input.shape[:-2]:
                batch_shape = gpytorch.utils.broadcasting._mul_broadcast_shape(batch_shape, input.shape[:-2])
                train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                input = input.expand(*batch_shape, *input.shape[-2:])
            full_inputs.append(torch.cat([train_input, input], dim=-2))
            ell_cond.append(self.spatial_covar_module.base_kernel.base_kernel.lengthscale_prior.conditional_sample(
                full_inputs[0][:,active_dims], given=(self.spatial_covar_module.base_kernel.inducing_points, 
                                                  torch.exp(self.log_ell_z)))
                           )
        full_output = self.forward(*full_inputs)
        full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix.evaluate_kernel()
        
        # Determine the shape of the joint distribution
        batch_shape = full_output.batch_shape
        joint_shape = full_output.event_shape
        tasks_shape = joint_shape[1:]  # For multitask learning
        test_shape = torch.Size([joint_shape[0] - self.train_inputs[0].shape[0], *tasks_shape])

        # Make the prediction
        test_mean = full_mean[..., train_inputs[0].shape[-2]:]
        test_test_covar = full_covar[..., train_inputs[0].shape[-2]:, train_inputs[0].shape[-2]:]
        test_train_covar = full_covar[..., train_inputs[0].shape[-2]:, :train_inputs[0].shape[-2]]

        # L is K_*z K_zz^{-1/2}^T i.e LL^T = K_*z K_zz^{-1} K_z*
        if isinstance(full_covar, gpytorch.lazy.LowRankRootLazyTensor):
            L = full_covar.root[..., train_inputs[0].shape[-2]:, :].evaluate()
        else:
            #L = full_covar._lazy_tensor.root[..., train_inputs[0].shape[-2]:, :].evaluate()
            L = full_covar.evaluate()[..., train_inputs[0].shape[-2]:, :]

        # A = K_{zz}^{-1/2} K_{zx} / \sigma 
        if isinstance(full_covar, gpytorch.lazy.LowRankRootLazyTensor):
            At = full_covar.root[..., :train_inputs[0].shape[-2], :].evaluate()/math.sqrt(self.likelihood.noise)
        else:
            At = full_covar.evaluate()[..., :train_inputs[0].shape[-2], :]/math.sqrt(
                self.likelihood.noise)
        # B = I + AA^T
        B = torch.eye(At.shape[-1]).to(At.device) + fn.t(At) @ At
        # mean = L B^{-1} A y / \sigma (+ mean_fn(x_*))
        predictive_mean = fn.mv(L, fn.mv(B, fn.mv(fn.t(At), self.train_targets), invert=True))/math.sqrt(
            self.likelihood.noise) + test_mean
        # covar = K_** - L (I - B^{-1}) L^\top
        predictive_covar = test_test_covar - (L @ ((
                            torch.eye(B.shape[-1]).to(B.device) 
                             - torch.inverse(B)
                                ) @ L.transpose(-1, -2)))
        # Reshape predictive mean to match the appropriate event shape
        predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
        return full_output.__class__(predictive_mean, predictive_covar)