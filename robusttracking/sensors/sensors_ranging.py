# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, unicode_literals)

from .actors import *
from .sensors import Sensor

import numpy as np
from numpy.linalg import norm,inv
from numpy import array,trace,diag,zeros,ones,full,eye,einsum

import jax.numpy as jnp

import torch

torch.set_default_tensor_type(torch.DoubleTensor)
ci = (1e-8)/3

eps = 1e-24
alpha = 0.000025
sigma_range2 = 25 #5*5


import jax.numpy as jnp


class RangingSensor(Sensor):
    
    def __init__(self,dim=3,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.e = zeros((dim,2*dim))
        self.e[:,:dim] = eye(dim)
    
    def distance(self,x,xs):
        v = (np.expand_dims(x,2)-np.expand_dims(xs,1)).swapaxes(0,1)
        return norm(self.e@v,axis=1)
    
    def ddistance(self,x,xs):
        v = (np.expand_dims(x,2)-np.expand_dims(xs,1)).swapaxes(0,1)
        denom_ = norm(self.e@v,axis=1,keepdims=True)
        denom = np.maximum(denom_,np.ones(denom_.shape)*eps)
        HJ = self.e@v/denom
        return HJ
    
    def hx(self,x,xs):
        '''mapping states to observations num_targets by num_sensors'''
        h = 2*ci*self.distance(x,xs)
        return np.maximum(h,np.ones(h.shape)*eps)
    
    def H(self,x,xs):
        '''Jocobian of hx at x'''
        v = (np.expand_dims(x,2)-np.expand_dims(xs,1)).swapaxes(0,1)
        denom_ = norm(self.e@v,axis=1,keepdims=True)
        denom = np.maximum(denom_,np.ones(denom_.shape)*eps)
        HJ = 2*ci*self.e@v/denom
        return HJ
    
    def sx(self,x,xs):
        '''mapping states to noise'''
        n_sensors = xs.shape[1]
        n_targets = x.shape[1]
        d = (1+alpha*self.distance(x,xs)**2)*(ci**2)*sigma_range2
        
        return np.einsum('ij,jk->ijk', d, np.eye(n_sensors))
    
    def S(self,x,xs):
        '''Jocobian of sx at x'''
        n_targets = x.shape[1]
        n_sensors = xs.shape[1]
        d = self.distance(x,xs)*(alpha*(ci**2)*sigma_range2)*2
        s = np.einsum('ij,jk->ijk', d, eye(n_sensors))
        dd = self.ddistance(x,xs)
        dcov = einsum('ijk,ibk->ibjk', s,dd)
        return dcov
    
class RangingSensorJ(Sensor):
    
    def __init__(self,dim=3,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        e = zeros((dim,2*dim))
        e[:,:dim] = eye(dim)
        self.e = jnp.array(e)
        
    def distance(self,x,xs):
        v = (jnp.expand_dims(x,2)-jnp.expand_dims(xs,1)).swapaxes(0,1)
        return jnp.linalg.norm(self.e@v,axis=1)
    
    def ddistance(self,x,xs):
        v = (jnp.expand_dims(x,2)-jnp.expand_dims(xs,1)).swapaxes(0,1)
        denom_ = jnp.linalg.norm(self.e@v,axis=1,keepdims=True)
        denom = jnp.maximum(denom_,jnp.ones(denom_.shape)*eps)
        HJ = self.e@v/denom
        return HJ
    
    def hx(self,x,xs):
        '''mapping states to observations num_targets by num_sensors'''
        h = 2*ci*self.distance(x,xs)
        return jnp.maximum(h,jnp.ones(h.shape)*eps)
    
    def H(self,x,xs):
        '''Jocobian of hx at x'''
        v = (jnp.expand_dims(x,2)-jnp.expand_dims(xs,1)).swapaxes(0,1)
        denom_ = jnp.linalg.norm(self.e@v,axis=1,keepdims=True)
        denom = jnp.maximum(denom_,jnp.ones(denom_.shape)*eps)
        HJ = 2*ci*self.e@v/denom
        return HJ
    
    def sx(self,x,xs):
        '''mapping states to noise'''
        n_sensors = xs.shape[1]
        n_targets = x.shape[1]
        d = (1+alpha*self.distance(x,xs)**2)*(ci**2)*sigma_range2
        
        return jnp.einsum('ij,jk->ijk', d, jnp.eye(n_sensors))
  
    def S(self,x,xs):
        '''Jocobian of sx at x'''
        n_targets = x.shape[1]
        n_sensors = xs.shape[1]
        d = self.distance(x,xs)*(alpha*(ci**2)*sigma_range2)*2
        s = jnp.einsum('ij,jk->ijk', d, jnp.eye(n_sensors))
        dd = self.ddistance(x,xs)
        dcov = jnp.einsum('ijk,ibk->ibjk', s,dd)
        return dcov
    
class RangingSensorT(Sensor): 
    def __init__(self,dim=3,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.e = torch.zeros((dim,2*dim))
        self.e[:,:dim] = torch.eye(dim)
        
    def distance(self,x,xs):
        v = (torch.unsqueeze(x,2)-torch.unsqueeze(xs,1)).swapaxes(0,1)
        return torch.linalg.norm(self.e@v,axis=1)
    
    def ddistance(self,x,xs):
        v = (torch.unsqueeze(x,2)-torch.unsqueeze(xs,1)).swapaxes(0,1)
        denom_ = torch.linalg.norm(self.e@v,axis=1,keepdims=True)
        denom = torch.maximum(denom_,torch.ones(denom_.shape)*eps)
        HJ = self.e@v/denom
        return HJ
    
    def hx(self,x,xs):
        '''mapping states to observations num_targets by num_sensors'''
        h = 2*ci*self.distance(x,xs)
        return torch.maximum(h,torch.ones(h.shape)*eps)
    
    def sx(self,x,xs):
        '''mapping states to noise'''
        n_sensors = xs.shape[1]
        n_targets = x.shape[1]
        s = torch.zeros((n_targets,n_sensors,n_sensors))
        d = (1+alpha*self.distance(x,xs)**2)*(ci**2)*sigma_range2
        return torch.diag_embed(d)
    
    # def sx(self,x,xs):
    #     '''mapping states to noise'''
    #     n_sensors = xs.shape[1]
    #     n_targets = x.shape[1]
    #     s = torch.zeros((n_targets,n_sensors,n_sensors))
    #     d = self.distance(x,xs)
    #     for i in range(n_targets):
    #         s[i] = torch.diag(1+alpha*(d[i]**2))
    #     s = s*(ci**2)*sigma_range2
    #     return s 
    
    def H(self,x,xs):
        '''Jocobian of hx at x'''
        v = (torch.unsqueeze(x,2)-torch.unsqueeze(xs,1)).swapaxes(0,1)
        denom_ = torch.linalg.norm(self.e@v,axis=1,keepdims=True)
        denom = torch.maximum(denom_,torch.ones(denom_.shape)*eps)
        HJ = 2*ci*self.e@v/denom
        return HJ

    # def S(self,x,xs):
    #     '''Jocobian of sx at x'''
    #     n_targets = x.shape[1]
    #     n_sensors = xs.shape[1]
    #     dcov = torch.zeros((n_targets,self.dim,n_sensors,n_sensors))
    #     d = self.distance(x,xs)
    #     dd = self.ddistance(x,xs)
    #     for i in range(n_targets):
    #         for j in range(self.dim):
    #             dcov[i,j] = torch.diag(2*d[i]*dd[i,j])
    #     dcov *= (alpha*(ci**2)*sigma_range2)
    #     return dcov
    def S(self,x,xs):
        '''Jocobian of sx at x'''
        n_targets = x.shape[1]
        n_sensors = xs.shape[1]
        dcov = torch.zeros((n_targets,self.dim,n_sensors,n_sensors))
        d = self.distance(x,xs)*(alpha*(ci**2)*sigma_range2)
        dd = self.ddistance(x,xs)
        dcov = torch.diag_embed(2*torch.bmm(dd,torch.diag_embed(d)))
        return dcov
    
    
    
class RangingSensor3DFull(RangingSensor,Linear3DFullActor):
    def __init__(self,n_static=0):
        super().__init__(dim=3,n_static=n_static)
        
class RangingSensor3DFullT(RangingSensorT,Linear3DFullActorT):
    def __init__(self,n_static=0):
        super().__init__(dim=3,n_static=n_static)
        
        
class RangingSensor3DFullJ(RangingSensorJ,Linear3DFullActorJ):
    def __init__(self,n_static=0):
        super().__init__(dim=3,n_static=n_static)
        
class RangingSensor3D2D(RangingSensor,Linear3D2DActor):
    def __init__(self,n_static=0):
        super().__init__(dim=3,n_static=n_static)
        
class RangingSensor3D2DT(RangingSensorT,Linear3D2DActorT):
    def __init__(self,n_static=0):
        super().__init__(dim=3,n_static=n_static)
        
class RangingSensor2D(RangingSensor,Linear2DActor):
    def __init__(self,n_static=0):
        super().__init__(dim=2,n_static=n_static)      
class RangingSensor2DT(RangingSensorT,Linear2DActorT):
    def __init__(self,n_static=0):
        super().__init__(dim=2,n_static=n_static)
class RangingSensor2DJ(RangingSensorJ,Linear2DActorJ):
    def __init__(self,n_static=0):
        super().__init__(dim=2,n_static=n_static)