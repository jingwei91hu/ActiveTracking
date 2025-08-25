# -*- coding: utf-8 -*-
from jax.numpy.linalg import inv,matrix_rank
from jax.numpy import trace,zeros,eye,einsum,transpose,pad

def bmm(A,B):
    return einsum('ijk,ika->ija',A,B)


def fisher_information_jnp(mu,cov,d_mu,d_cov):
    inv_cov = inv(cov)
    I = bmm(bmm(d_mu,inv_cov),transpose(d_mu,(0,2,1)))
    I3 = einsum('aik, ackj -> acij', inv_cov, d_cov)
    I4 = einsum('abik, ackj -> abc', I3, I3)
    return I+0.5*I4   


def crlb_jnp(mu,cov,d_mu,d_cov,W=None,c_inv_hat = None):
    f = fisher_information_jnp(mu,cov,d_mu,d_cov)
    
    if c_inv_hat is None:
        c = inv(f)
    elif c_inv_hat.shape[-1]!=f.shape[-1]:
        paddings = c_inv_hat.shape[-1] - f.shape[-1]
        c = inv(pad(f,[(0,0),(0,paddings),(0,paddings)]) + c_inv_hat)
    else:
        c = inv(f + c_inv_hat)
    if W is None:
        return einsum("...ii",c).mean()
    return einsum("...ii",bmm(W,c)).mean()
