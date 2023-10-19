#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience functions, mainly for abstracting torch's linear algebra with terser names and better batching.
"""

from typing import Optional
import math
import torch
import bisect
from collections import namedtuple


def dot(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Compute the batch dot product v1^T v2"""
    return (v1 * v2).sum(-1)


def t(x: torch.Tensor) -> torch.Tensor:
    """Matrix transpose"""
    return torch.transpose(x, -1, -2)


def tr(x: torch.Tensor) -> torch.Tensor:
    """Trace"""
    return torch.diagonal(x, dim1=-1, dim2=-2).sum(-1)


def mv(matrix: torch.Tensor, vector: torch.Tensor, invert=False) -> torch.Tensor:
    if invert is False:
        return torch.squeeze(matrix @ torch.unsqueeze(vector, -1), -1)
    else:
        return torch.linalg.solve(matrix, vector.unsqueeze(-1)).squeeze(-1)

def quad(v: torch.Tensor, matrix: torch.Tensor, v2: torch.Tensor = None, invert: bool = False) -> torch.Tensor:
    if v2 is None:
        v2 = v.clone()
    if invert == True:
        # WARNING: need to add a final dimension to v2 for it to be treated as a vector, contrary to docs
        v2 = torch.linalg.solve(matrix, v2.unsqueeze(-1)).squeeze(-1)
    else:
        v2 = mv(matrix, v2)
    return dot(v, v2)

def expquad(
    v: torch.Tensor,
    matrix: torch.Tensor,
    invert: bool = False,
    out_scale: torch.Tensor = torch.tensor(1.0),
    exp_scale: torch.tensor = torch.tensor(0.5),
) -> torch.Tensor:
    return out_scale * torch.exp(-exp_scale * quad(v, matrix, invert=invert))


def sym(x: torch.Tensor) -> torch.Tensor:
    """Force symmetry"""
    return 0.5 * (x + t(x))


def op(v1: torch.Tensor, v2: Optional[torch.Tensor]=None) -> torch.Tensor:
    """Vector outer product"""
    if v2 is None:
        v2 = v1
    return v1.unsqueeze(-1) @ v2.unsqueeze(-2)


def bisect_right(array, value, tol=1e-8):
    """Bisect right which is robust up to a tolerance. Returns the index i to insert
    value in (sorted) array such that array[j] <= value+tol for j < i, array[j] > value+tol
    for j >= i. tol should be smaller than gaps between values, and used to avoid issues with
    floating points only."""
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    return bisect.bisect_right(array, value + tol)


def bisect_left(array, value, tol=1e-8):
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    return bisect.bisect_left(array, value + tol)


def vec(x: torch.Tensor):
    """Vectorises a matrix"""
    batch_shape = x.shape[:-2]
    return t(x).contiguous().view(*batch_shape, x.shape[-2] * x.shape[-1])


def vech(x: torch.Tensor):
    """Half vectorisation of a matrix, i.e. vectorises the lower triangle"""
    D = x.shape[-2]
    if not x.shape[-1] == D:
        raise ValueError("Matrix must be square for half vectorisation, but got shape {}".format(x.shape))
    return x[..., torch.tril(torch.ones(D, D)) == 1]


def kron(x: torch.Tensor, y: torch.Tensor):
    """Batch kronecker product of matrices"""
    size1 = torch.Size(torch.tensor(x.shape[-2:]) * torch.tensor(y.shape[-2:]))
    res = x.unsqueeze(-1).unsqueeze(-3) * y.unsqueeze(-2).unsqueeze(-4)
    size0 = res.shape[:-4]
    return res.reshape(size0 + size1)


def duplication_matrix(n: int):
    out = torch.zeros(n ** 2, n * (n + 1) // 2)
    for j in range(1, n + 1):
        for i in range(j, n + 1):
            u = torch.zeros(n * (n + 1) // 2)
            u[(j - 1) * n + i - j * (j - 1) // 2 - 1] = 1.0
            T = torch.zeros(n, n)
            T[i - 1, j - 1] = 1.0
            T[j - 1, i - 1] = 1.0
            out = out + fn.op(vec(T), u)
    return out


def diff(x, boundary_value=None, dim=-2):
    """Return the forward differences (output[...,n] is x[...,n+1]-x[...,n]). To retain length, the final value should be
    given, else it will be replaced by replicating the penultimate."""
    x = x.transpose(dim, -1)
    diff_x = (x[..., 1:] - x[..., :-1]).view(*x.shape[:-1], x.shape[-1] - 1)
    if boundary_value is None:
        boundary_value = diff_x[..., -1]
    return torch.cat((diff_x, boundary_value.unsqueeze(-1)), dim=-1).transpose(dim, -1)


def normalise(x: torch.Tensor, **kwargs):
    """x is normalised to be zero mean and unit norm over the last dimension. Extra kwargs are passed to
    pytorch's normalize to e.g. change the norm, dimension, ..."""
    if "dim" in kwargs.keys():
        dim = kwargs["dim"]
    else:
        dim = -1
    x = x - torch.mean(x, dim=dim, keepdim=True)
    return math.sqrt(x.shape[dim]) * torch.nn.functional.normalize(x, **kwargs)


def project_pca(D: int, y: torch.Tensor):
    """For y (..., N, Delta), project y onto the D directions of maximum variance"""
    eigenvals, eigenvecs = torch.linalg.eigh(t(y) @ y)
    Result = namedtuple("result", "projection matrix")
    return Result(mv(t(eigenvecs[..., -D:]), y), t(eigenvecs[..., -D:]))


def project_lstsq(y: torch.Tensor, C: torch.Tensor, d: torch.Tensor):
    """Assuming y = Cx+d, return the least squares solution for x."""
    soln = torch.linalg.lstsq(C, t(y - d))
    Result = namedtuple("result", "projection matrix")
    return Result(t(soln.solution), C)

def robust_logdet(x : torch.Tensor, init_scale=1e-30, max_scale = 1e-6) -> torch.Tensor:
    """Does logdet of batched matrices, adding diagonal terms until max_scale is reached, or no nan values.
    
    The diagonal matrix is a scaled identity matrix added to all batches."""
    out = torch.logdet(x)
    if torch.any(torch.isnan(out)):
        reg = init_scale * torch.eye(x.shape[-1]).to(x.device)
        while reg[0,0].squeeze() <= max_scale and torch.any(torch.isnan(out)):
            out = torch.logdet(x + reg)
            reg = reg * 10
    return out
