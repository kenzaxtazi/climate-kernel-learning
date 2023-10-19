#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GP models using Gibbs kernels.
Classes
-------
DiagonalExactGP(train_x, train_y, likelihood, prior, num_dim : int=1)
    (Single output, num_dim input GP)
    Works like gpytorch.models.ExactGP, except that the forward call should only be used on training data (for now)
    and predictions should be made by calling model.predict(x_new). Uses zero mean and scaled Gibbs covariance function.
DiagonalSparseGP(train_x, train_y, likelihood, prior, z, num_dim : int=1)
    (Single output, num_dim input GP)
    Works like gpytorch.models.ExactGP, except that predictions should be made using model.predict(x_new), and the 
    predictions currently only return the correct marginal covariances. Uses a scaled Gibbs covariance function with
    inducing points.
"""
import math
import torch
import gpytorch
from models.gibbs_kernels import GibbsKernel, GibbsSafeScaleKernel, InducingGibbsKernel
from utils import functional as fn

class DiagonalExactGP(gpytorch.models.ExactGP):
    """Model for MAP inference of diagonal Gibbs kernel GP. 
    """
    def __init__(self, train_x, train_y, likelihood, prior, num_dim=1):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = GibbsSafeScaleKernel(
                GibbsKernel(lengthscale_prior=prior, ard_num_dims=num_dim, 
                                        ))
        self.register_parameter('log_ell_train_x', 
                                torch.nn.Parameter(
                                    self.covar_module.base_kernel.lengthscale_prior.forward(train_x).mean.clone()
                                ))
        self.register_prior('ell_train_prior', 
                            self.covar_module.base_kernel.lengthscale_prior, 
                            lambda module : (module.train_inputs[0], module.log_ell_train_x)
                           )
        
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x, ell1=torch.exp(self.log_ell_train_x))
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
    def predict(self, x_new):
        """Returns predictive at x_new given estimate of lengthscales at training points.
        This should really be a modified prediction strategy, but this will do for now."""
        K_xx = self.covar_module(self.train_inputs[0], ell1=torch.exp(self.log_ell_train_x)).evaluate()
        ell2 = self.covar_module.base_kernel.lengthscale_prior.conditional_sample(x_new, given=(self.train_inputs[0], 
                                            torch.exp(self.log_ell_train_x)))
        K_ss = self.covar_module(x_new, ell1=ell2).evaluate()
        K_sx = self.covar_module(x_new, self.train_inputs[0], ell1=ell2, 
                                ell2=torch.exp(self.log_ell_train_x)).evaluate()

        mu = fn.dot(K_sx, fn.mv(K_xx + (self.likelihood.noise) * torch.eye(K_xx.shape[-1]).to(K_xx.device),
                        self.train_targets, invert=True))
        sigma =  K_ss - K_sx @ torch.inverse(K_xx + (self.likelihood.noise) 
                                            * torch.eye(K_xx.shape[-1]).to(K_xx.device)) @ fn.t(K_sx)

        f_pred = gpytorch.distributions.MultivariateNormal(mu, 
                                                covariance_matrix=sigma + 1e-4*torch.eye(K_ss.shape[-1]).to(K_ss.device))
        return f_pred

class DiagonalSparseGP(gpytorch.models.ExactGP):
    """Model for MAP inference of sparse Gibbs kernel GP.
    """
    def __init__(self, train_x, train_y, likelihood, prior, z, num_dim=1):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = GibbsSafeScaleKernel(
                            InducingGibbsKernel(
            GibbsKernel(lengthscale_prior=prior, ard_num_dims=num_dim),
                            z, likelihood)
        )
            
        self.register_parameter('log_ell_z', 
                                torch.nn.Parameter(
                                    self.covar_module.base_kernel.base_kernel.lengthscale_prior.forward(z).mean.clone()
                                ))
        self.register_prior('ell_z_prior', 
                            self.covar_module.base_kernel.base_kernel.lengthscale_prior, 
                            lambda module : (module.covar_module.base_kernel.inducing_points, module.log_ell_z)
                           )
        
    def forward(self, x, ell=None):
        mean = self.mean_module(x)
        covar = self.covar_module(x, ell=torch.exp(self.log_ell_z))

        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
    def predict(self, x_new):
        """Returns predictive at x_new given estimate of lengthscales at inducing points.
        WARNING: currently only the marginals of this are correct"""
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [x_new.unsqueeze(-1) if x_new.ndimension() == 1 else x_new]
        
        # Concatenate the input to the training input
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
            ell_cond.append(self.covar_module.base_kernel.base_kernel.lengthscale_prior.conditional_sample(
                full_inputs[0], given=(self.covar_module.base_kernel.inducing_points, 
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
            L = full_covar._lazy_tensor.root[..., train_inputs[0].shape[-2]:, :].evaluate()
        # A = K_{zz}^{-1/2} K_{zx} / \sigma 
        if isinstance(full_covar, gpytorch.lazy.LowRankRootLazyTensor):
            At = full_covar.root[..., :train_inputs[0].shape[-2], :].evaluate()/math.sqrt(self.likelihood.noise)
        else:
            At = full_covar._lazy_tensor.root[..., :train_inputs[0].shape[-2], :].evaluate()/math.sqrt(
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