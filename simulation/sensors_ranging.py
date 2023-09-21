# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, unicode_literals)

from .actors import *
from .sensors import Sensor

import numpy as np
from numpy.linalg import norm,inv
from numpy import array,trace,diag,zeros,ones,full

import torch


torch.set_default_tensor_type(torch.DoubleTensor)
ci = (1e-8)/3

eps = 1e-20
alpha = 0.01

class RangingSensor(Sensor):
    
    def __init__(self,dim=3,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.e = np.zeros((dim,2*dim))
        self.e[:,:dim] = np.eye(dim)
        
    def hx(self,x,xs):
        '''mapping states to observations num_targets by num_sensors'''
        v = (np.expand_dims(x,2)-np.expand_dims(xs,1)).swapaxes(0,1)
        return 2*ci*norm(self.e@(v+eps),axis=1)
    
    def H(self,x,xs):
        '''Jocobian of hx at x'''
        v = (np.expand_dims(x,2)-np.expand_dims(xs,1)).swapaxes(0,1)
        denom = norm(self.e@(v+eps),axis=1,keepdims=True)
        HJ = 2*ci*self.e@v/denom
        return HJ
    
    def sx(self,x,xs):
        '''mapping states to noise'''
        n_sensors = xs.shape[1]
        n_targets = x.shape[1]
        s = zeros((n_targets,n_sensors,n_sensors))
        d = self.hx(x,xs)
        for i in range(n_targets):
            s[i] = diag(1+alpha*d[i])
        s = s*(ci**2)
        return s
    
    def S(self,x,xs):
        '''Jocobian of sx at x'''
        n_targets = x.shape[1]
        n_sensors = xs.shape[1]
        dcov = zeros((n_targets,self.dim,n_sensors,n_sensors))
        dmu = self.H(x,xs)
        for i in range(n_targets):
            for j in range(self.dim):
                dcov[i,j] = diag(dmu[i,j])
        dcov *= (alpha*(ci**2))
        return dcov
    
class RangingSensorT(Sensor): 
    def __init__(self,dim=3,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.e = torch.zeros((dim,2*dim))
        self.e[:,:dim] = torch.eye(dim)
        
    def hx(self,x,xs):
        '''mapping states to observations num_targets by num_sensors'''
        v = (torch.unsqueeze(x,2)-torch.unsqueeze(xs,1)).swapaxes(0,1)
        return 2*ci*torch.linalg.norm(self.e@(v+eps),axis=1)
    
    def sx(self,x,xs):
        '''mapping states to noise'''
        n_sensors = xs.shape[1]
        n_targets = x.shape[1]
        s = torch.zeros((n_targets,n_sensors,n_sensors))
        d = self.hx(x,xs)
        for i in range(n_targets):
            s[i] = torch.diag(1+alpha*d[i])
        s = s*(ci**2)
        return s
    
    def H(self,x,xs):
        '''Jocobian of hx at x'''
        v = (torch.unsqueeze(x,2)-torch.unsqueeze(xs,1)).swapaxes(0,1)
        denom = torch.linalg.norm(self.e@(v+eps),axis=1,keepdims=True)
        HJ = 2*ci*self.e@v/denom
        return HJ
    
    def S(self,x,xs):
        '''Jocobian of sx at x'''
        n_targets = x.shape[1]
        n_sensors = xs.shape[1]
        dcov = torch.zeros((n_targets,self.dim,n_sensors,n_sensors))
        dmu = self.H(x,xs)
        for i in range(n_targets):
            for j in range(self.dim):
                dcov[i,j] = torch.diag(dmu[i,j])
        dcov *= (alpha*(ci**2))
        return dcov
    
class RangingSensor3DFull(RangingSensor,Linear3DFullActor):
    def __init__(self,n_static=0):
        super().__init__(dim=3,n_static=n_static)
        
class RangingSensor3DFullT(RangingSensorT,Linear3DFullActorT):
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
    
__all__=['RangingSensor3DFull','RangingSensor3DFullT','RangingSensor3D2D','RangingSensor3D2DT','RangingSensor2D','RangingSensor2DT']