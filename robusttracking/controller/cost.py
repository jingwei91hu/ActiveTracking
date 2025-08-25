# -*- coding: utf-8 -*-
import torch
from torch.linalg import inv,matrix_rank
from torch import trace,zeros,eye,einsum,transpose,bmm,inverse
from torch.nn.functional import pad

# def fisher_information_torch(mu,cov,d_mu,d_cov):
#     n_targets,dim,n_sensors = d_mu.shape
#     I = zeros((n_targets,dim,dim))
#     for k in range(n_targets):
#         inv_cov = inv(cov[k])
#         for i in range(dim):
#             for j in range(dim):
#                 I[k,i,j] = d_mu[k,[i],:]@inv_cov@d_mu[k,[j],:].T + 0.5*trace(inv_cov@d_cov[k,i,:,:]@inv_cov@d_cov[k,j,:,:])
#     return I

def fisher_information_torch(mu,cov,d_mu,d_cov):
    inv_cov = inverse(cov)
    I = bmm(bmm(d_mu,inv_cov),transpose(d_mu,1,2))
    I3 = einsum('aik, ackj -> acij', inv_cov, d_cov)
    I4 = einsum('abik, ackj -> abc', I3, I3)
    return I+0.5*I4


# def crlb_torch(mu,cov,d_mu,d_cov,W,c_inv_hat = None):
#     f = fisher_information_torch(mu,cov,d_mu,d_cov)
#     n_targets,dim,dim = f.shape
    
#     J = 0
#     for i in range(n_targets):
#         if c_inv_hat==None:
#             lb = inv(f[i])
#         else:
#             fi = zeros(c_inv_hat[i].shape)
#             fi[:,:] += c_inv_hat[i]
#             fi[:dim,:dim] += f[i]
#             lb = inv(fi)
            
#         J += trace(W@lb)
        
#     return J

def crlb_torch(mu,cov,d_mu,d_cov,W=None,c_inv_hat = None):
    f = fisher_information_torch(mu,cov,d_mu,d_cov)
    
    if c_inv_hat==None:
        c = inverse(f)
    elif c_inv_hat.shape[-1]!=f.shape[-1]:
        paddings = c_inv_hat.shape[-1] - f.shape[-1]
        c = inverse(pad(f,(0,paddings,0,paddings)) + c_inv_hat)
    else:
        c = inverse(f + c_inv_hat)
    
    if W==None:
        return einsum("...ii",c).mean()
    return einsum("...ii",bmm(W,c)).mean()


# def barrier_cost(xs,S,lb,ub):
#     K,D,M = xs.shape
#     Je = 0
#     for k in range(K):
#         for m in range(M):
#             s = S@xs[k][:,[m]]
#             Je += ((torch.relu(lb - s) + torch.relu(s - ub))**2).sum()
#     return Je/K/M

def barrier_cost(xs,S,lb,ub):
    s = S@xs
    Je = ((torch.relu(lb-s)+torch.relu(s - ub))**2).sum()
    return Je