#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gibbs kernel, and associated prior classes and wrapper kernels.
Classes
-------
PositivePriorProcess()
    Very minimal base class for prior processes. Note forward returns some distribution depending on the 
    parameterisation (the distribution of the unconstrained value), but sample and conditional_sample return the actual 
    (positive) value.
LogNormalPriorProcess(input_dim : int=1, covariance_function : gpytorch.kernel.Kernel=None)
    Places a GP prior process on the log-value. In multiple dimensions, an independent GP is used for each dim.
    Currently conditional_sample just returns a single sample at the conditional mean.
    Default kernel is a scaled RBF kernel.
GibbsKernel(*kernel_args, lengthscale_prior : PositivePriorProcess=None, **kernel_kwargs)
    Basic diagonal Gibbs kernel, with default lengthscale prior as a LogNormalPriorProcess. The forward method expects
    a lengthscale (ell1) to be given, else it will resample it. If the the inputs x1, x2 are different and only one
    lengthscale is given, the other will be sampled conditioned on ell1. Use with models.nonstationary.DiagonalExactGP.
GibbsSafeScaleKernel(*kernel_args, **kernel_kwargs)
    Scale wrapper for use with GibbsKernel. The defualt behaviour of kernels takes the batch shape from any subkernel;
    this would include the kernel in the lengthscale prior, so we need a new class where we set the batch shape
    explicitly.
InducingGibbsKernel(base_kernel : GibbsKernel, inducing_points : torch.Tensor,
                    likelihood : gpytorch.likelihoods.Likelihood, active_dims : Optional[Tuple[int, ...]]=None)
    Wrapper adapted from InducingPointKernel for GibbsKernel. The key difference is when calling you should pass
    ell as the lengthscale at the inducing points, then the lengthscales at train/test points will be correcly (jointly)
    sampled conditional on these. Use with models.nonstationary.DiagonalSparseGP.
"""

import math
import torch
import gpytorch
from typing import Tuple, Optional
from utils import functional as fn

class PositivePriorProcess(torch.nn.Module):
    """Base class for lengthscale prior processes."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        
    def forward(self,
               x : torch.Tensor) -> torch.distributions.Distribution:
        """Returns some distribution; depends on the parameterisation"""
        raise NotImplementedError
    
    def sample(self,
              x : torch.Tensor, 
               **kwargs) -> torch.Tensor:
        """Returns sampled positive vector (..., D,) or positive matrix (..., D, D,) at input locations x"""
        raise NotImplementedError
    
    def conditional_sample(self,
                           x : torch.Tensor,
                           given : Tuple[torch.Tensor, torch.Tensor],
                           **kwargs) -> torch.Tensor:
        """Returns sampled positive vector (..., D,) or positive matrix (..., D, D,) at input locations x,
        conditioned on the process being given[1] at inputs given[0]."""
        raise NotImplementedError
        
class LogNormalPriorProcess(PositivePriorProcess):
    """D independent GPs for the log-value."""
    def __init__(self, input_dim : int=1, covariance_function : gpytorch.kernels.Kernel=None, active_dims=None) -> None:
        super().__init__()
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size((input_dim,)))
        if covariance_function is None:
            covariance_function = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
                ard_num_dims=input_dim, batch_shape=torch.Size((input_dim,)), active_dims=active_dims), batch_shape=torch.Size((input_dim,)), active_dims=active_dims
            )
        self.covar_module = covariance_function
        
    def forward(self, x : torch.Tensor) -> torch.distributions.MultivariateNormal:
        """Returns log-value distribution"""
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x),
                                                        self.covar_module(x))
    
    def sample(self, x : torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.exp(self.forward(x).rsample(**kwargs))
    
    def conditional_sample(self, x : torch.Tensor, 
                           given : Tuple[torch.Tensor, torch.Tensor],
                           **kwargs) -> torch.Tensor:
        prior_dist_given = self(given[0]) 
        prior_x = self.forward(x)
        K_xg = self.covar_module(x, given[0]).evaluate()
        K_xg = K_xg.permute(*range(1, len(K_xg.shape)-1), 0, -1)
        prior_mean = prior_x.mean.permute(*range(1, len(prior_x.mean.shape)), 0)    
        jitter = 1e-4 * torch.eye(given[0].shape[-2])
        mu = prior_mean + fn.dot(K_xg, fn.mv(prior_dist_given.covariance_matrix + jitter,
                                        torch.log(given[1]) - prior_dist_given.mean,
                                        invert=True
                                        )
                                  )
        # for one sample just return the conditional mean.
        #sigma = (prior_x.covariance_matrix 
        #         - K_xg @ torch.inverse(prior_dist_given.covariance_matrix + jitter) @ fn.t(K_xg)
        #        + 1e-5 * torch.eye(x.shape[-1]))
       
        #return torch.exp(torch.distributions.MultivariateNormal(mu, sigma).rsample(**kwargs))
        return torch.exp(mu).permute(-1, *range(0, len(mu.shape)-1))
    
    def log_prob(self, x_and_logell : Tuple[torch.Tensor, torch.Tensor]):
        """returns log probability of output"""
        x, log_value = x_and_logell
        #log_value = torch.log(ell)
        mu = self.mean_module(x)
        sigma = self.covar_module(x) + 1e-4 * torch.eye(x.shape[-2]).to(x.device)
        out = gpytorch.distributions.MultivariateNormal(mu, sigma).log_prob(log_value)
        return out/x.shape[-2]

class GibbsKernel(gpytorch.kernels.Kernel):
    
    is_stationary = False
    
    def __init__(self,
                 *args,
                 lengthscale_prior : PositivePriorProcess=None,
                 **kwargs,
                ) -> None:
        """Gibbs kernel of eq 4.32 in Rasmussen & Williams.
        
        lengthscale_prior : PositiveProcess
            Should have methods to get the joint dist of (D,) or (D,D) lengthscales at each point,
            and conditional samples.
        """
        super().__init__(*args, **kwargs)
        self.lengthscale_prior = lengthscale_prior
        

    @property
    def batch_shape(self):
        """Overriding normal behaviour to avoid setting batch shape based on prior's kernels."""
        return self._batch_shape
    
    def forward(self,
                x1 : torch.Tensor,
                x2 : torch.Tensor,
                ell1 : Optional[torch.Tensor] = None,
                ell2 : Optional[torch.Tensor] = None,
                **kwargs
               ) -> torch.Tensor:
        """Computes the Gibbs kernel matrix. ell1, ell2 are assumed to be the kernel lengthscales at
        x1 and x2 respectively. If only ell1 is given, and x1 =/= x2, then ell2 is sampled conditionally."""
        if ell1 is None:
            ell1 = self.lengthscale_prior.sample(x1)
            self.ell1 = ell1
        
        if torch.equal(x1, x2):
            ell2 = ell1
        else:
            if ell2 is None:
                ell2 = self.lengthscale_prior.conditional_sample(x2, given=(x1, ell1))
                self.ell2 = ell2
        sq_sum = (ell1.unsqueeze(-1)**2 + ell2.unsqueeze(-2)**2)
        out = torch.sqrt(2 * fn.op(ell1, ell2) / sq_sum)
        out = torch.prod(out, dim=-3) # (..., n1, n2)
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        out = out * torch.exp(- 
                    torch.sum(diff**2 / sq_sum.permute(*range(len(sq_sum.shape)-3), -2, -1, -3),
                             dim=-1)
                             )
        return out

class GibbsSafeScaleKernel(gpytorch.kernels.ScaleKernel):
    @property
    def batch_shape(self):
        """Overriding normal behaviour to avoid setting batch shape based on prior's kernels."""
        return self._batch_shape


class InducingGibbsKernel(gpytorch.kernels.InducingPointKernel):
    """Wrapper for Gibbs Kernel for SGPR."""
    def __init__(
        self,
        base_kernel : GibbsKernel,
        inducing_points : torch.Tensor,
        likelihood : gpytorch.likelihoods.Likelihood,
        active_dims : Optional[Tuple[int,...]] = None,
    ):
        super().__init__(base_kernel, inducing_points, likelihood, active_dims)
    
    @property
    def batch_shape(self):
        """Overriding normal behaviour to avoid setting batch shape based on prior's kernels."""
        return self._batch_shape
        
    def _inducing_mat(self, ell=None):
        if not self.training and hasattr(self, "_cached_kernel_mat"):
            return self._cached_kernel_mat
        else:
            res = gpytorch.lazy.delazify(self.base_kernel(self.inducing_points, self.inducing_points,
                                                         ell1=ell))
            if not self.training:
                self._cached_kernel_mat = res
            return res
    
    def _inducing_inv_root(self, ell=None):
        if not self.training and hasattr(self, "_cached_kernel_inv_root"):
            return self._cached_kernel_inv_root
        else:
            chol = gpytorch.utils.cholesky.psd_safe_cholesky(self._inducing_mat(ell), upper=True)
            eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
            inv_root = torch.triangular_solve(eye, chol)[0]

            res = inv_root
            if not self.training:
                self._cached_kernel_inv_root = res
            return res
    
    def _get_covariance(self, x1, x2, ell):
        # first sample lengthscales
        if torch.equal(x1, x2):
            ell1 = self.base_kernel.lengthscale_prior.conditional_sample(x1, given=(self.inducing_points,
                                                                                    ell))
            ell2 = ell1
        else:
            ell_cond = self.base_kernel.lengthscale_prior.conditional_sample(torch.cat((x1, x2), dim=-2), 
                                                    given=(self.inducing_points, ell))
            ell1 = ell_cond[...,:x1.shape[-2], :]
            ell2 = ell_cond[...,x1.shape[-2]:, :]

        k_ux1 = gpytorch.lazy.delazify(self.base_kernel(x1, self.inducing_points, ell1=ell1,
                                                       ell2=ell))
        if torch.equal(x1, x2):
            covar = gpytorch.lazy.LowRankRootLazyTensor(k_ux1.matmul(self._inducing_inv_root(ell)))

            # Diagonal correction for predictive posterior
            if not self.training and gpytorch.settings.sgpr_diagonal_correction.on():
            #    # TODO: sort out lengthscale sampling here
                correction = (self.base_kernel(x1, x2, diag=True, ell1=ell1, ell2=ell2) - covar.diag()).clamp(0, math.inf)
                covar = gpytorch.lazy.LowRankRootAddedDiagLazyTensor(covar, 
                                                        gpytorch.lazy.DiagLazyTensor(correction))
        else:
            k_ux2 = gpytorch.lazy.delazify(self.base_kernel(x2, self.inducing_points, ell1=ell2, ell2=ell))
            covar = gpytorch.lazy.MatmulLazyTensor(
                k_ux1.matmul(self._inducing_inv_root(ell)), k_ux2.matmul(self._inducing_inv_root(ell)).transpose(-1, -2)
            )

        return covar, ell1, ell2

    def _covar_diag(self, inputs, ell):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        # Get diagonal of covar
        covar_diag = gpytorch.lazy.delazify(self.base_kernel(inputs, diag=True, ell1=ell))
        return gpytorch.lazy.DiagLazyTensor(covar_diag)
        
    def forward(self, x1, x2, diag=False, ell=None, **kwargs):
        covar, ell1, ell2 = self._get_covariance(x1, x2, ell=ell)

        if self.training:
            if not torch.equal(x1, x2):
                raise RuntimeError("x1 should equal x2 in training mode")
            zero_mean = torch.zeros_like(x1.select(-1, 0))
            new_added_loss_term = gpytorch.mlls.InducingPointKernelAddedLossTerm(
                gpytorch.distributions.MultivariateNormal(zero_mean, self._covar_diag(x1, ell1)),
                gpytorch.distributions.MultivariateNormal(zero_mean, covar),
                self.likelihood,
            )
            self.update_added_loss_term("inducing_point_loss_term", new_added_loss_term)

        if diag:
            return covar.diag()
        else:
            return covar

class InducingGibbsKernelST(gpytorch.kernels.InducingPointKernel):
    """Wrapper for Gibbs Kernel for SGPR."""
    def __init__(
        self,
        base_kernel : GibbsKernel,
        inducing_points : torch.Tensor,
        likelihood : gpytorch.likelihoods.Likelihood,
        active_dims : Optional[Tuple[int,...]] = None,
    ):
        super().__init__(base_kernel, inducing_points, likelihood, active_dims)
    
    @property
    def batch_shape(self):
        """Overriding normal behaviour to avoid setting batch shape based on prior's kernels."""
        return self._batch_shape
        
    def _inducing_mat(self, ell=None):
        if not self.training and hasattr(self, "_cached_kernel_mat"):
            return self._cached_kernel_mat
        else:
            res = gpytorch.lazy.delazify(self.base_kernel(self.inducing_points[:,self.active_dims], self.inducing_points[:,self.active_dims],
                                                         ell1=ell))
            if not self.training:
                self._cached_kernel_mat = res
            return res
    
    def _inducing_inv_root(self, ell=None):
        if not self.training and hasattr(self, "_cached_kernel_inv_root"):
            return self._cached_kernel_inv_root
        else:
            chol = gpytorch.utils.cholesky.psd_safe_cholesky(self._inducing_mat(ell), upper=True)
            eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
            inv_root = torch.triangular_solve(eye, chol)[0]

            res = inv_root
            if not self.training:
                self._cached_kernel_inv_root = res
            return res
    
    def _get_covariance(self, x1, x2, ell):
        # first sample lengthscales
        if torch.equal(x1, x2):
            ell1 = self.base_kernel.lengthscale_prior.conditional_sample(x1, given=(self.inducing_points[:,self.active_dims],
                                                                                    ell))
            ell2 = ell1
        else:
            ell_cond = self.base_kernel.lengthscale_prior.conditional_sample(torch.cat((x1, x2), dim=-2), 
                                                    given=(self.inducing_points[:,self.active_dims], ell))
            ell1 = ell_cond[...,:x1.shape[-2], :]
            ell2 = ell_cond[...,x1.shape[-2]:, :]

        k_ux1 = gpytorch.lazy.delazify(self.base_kernel(x1, self.inducing_points[:,self.active_dims], ell1=ell1,
                                                       ell2=ell))
        if torch.equal(x1, x2):
            covar = gpytorch.lazy.LowRankRootLazyTensor(k_ux1.matmul(self._inducing_inv_root(ell)))

            # Diagonal correction for predictive posterior
            if not self.training and gpytorch.settings.sgpr_diagonal_correction.on():
            #    # TODO: sort out lengthscale sampling here
                correction = (self.base_kernel(x1, x2, diag=True, ell1=ell1, ell2=ell2) - covar.diag()).clamp(0, math.inf)
                covar = gpytorch.lazy.LowRankRootAddedDiagLazyTensor(covar, 
                                                        gpytorch.lazy.DiagLazyTensor(correction))
        else:
            k_ux2 = gpytorch.lazy.delazify(self.base_kernel(x2, self.inducing_points[:,self.active_dims], ell1=ell2, ell2=ell))
            covar = gpytorch.lazy.MatmulLazyTensor(
                k_ux1.matmul(self._inducing_inv_root(ell)), k_ux2.matmul(self._inducing_inv_root(ell)).transpose(-1, -2)
            )

        return covar, ell1, ell2

    def _covar_diag(self, inputs, ell):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        # Get diagonal of covar
        covar_diag = gpytorch.lazy.delazify(self.base_kernel(inputs, diag=True, ell1=ell))
        return gpytorch.lazy.DiagLazyTensor(covar_diag)
        
    def forward(self, x1, x2, diag=False, ell=None, **kwargs):
        covar, ell1, ell2 = self._get_covariance(x1, x2, ell=ell)

        if self.training:
            if not torch.equal(x1, x2):
                raise RuntimeError("x1 should equal x2 in training mode")
            zero_mean = torch.zeros_like(x1.select(-1, 0))
            new_added_loss_term = gpytorch.mlls.InducingPointKernelAddedLossTerm(
                gpytorch.distributions.MultivariateNormal(zero_mean, self._covar_diag(x1, ell1)),
                gpytorch.distributions.MultivariateNormal(zero_mean, covar),
                self.likelihood,
            )
            self.update_added_loss_term("inducing_point_loss_term", new_added_loss_term)

        if diag:
            return covar.diag()
        else:
            return covar

