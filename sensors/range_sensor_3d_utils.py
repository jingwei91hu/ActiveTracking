# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, unicode_literals)

import numpy as np
from numpy.linalg import norm,inv
from numpy import array,trace,diag,zeros,ones,full

import torch



torch.set_default_tensor_type(torch.DoubleTensor)
ci = (1e-8)/3
ee = array([[1.,0.,0.,0., 0., 0.]
                ,[0.,1.,0., 0., 0.,0]
                ,[0.,0, 1.,0., 0., 0.]])

FF = array([[1., 0., 0., 1., 0., 0.]
                 ,[0., 1., 0., 0., 1., 0.]
                 ,[0., 0., 1., 0., 0., 1.]
                 ,[0., 0., 0., 1., 0., 0.]
                 ,[0., 0., 0., 0., 1., 0.]
                 ,[0., 0., 0., 0., 0., 1.]])

BBT = array([[0. , 0. , 0., 0.5, 0., 0.]
                 ,[0. , 0. , 0. , 0., 0.5, 0.]
                 ,[0. , 0. , 0. , 0., 0., 0.5]
                 ,[0. , 0. , 0. ,1. , 0.,  0. ]
                 ,[0. , 0., 0. , 0.  , 1., 0. ]
                 ,[0. , 0., 0. , 0.  , 0,  1. ]])

BBS = array([[0. , 0. , 0., 0.5, 0., 0.]
                 ,[0. , 0. , 0. , 0., 0.5, 0.]
                 ,[0. , 0. , 0. , 0., 0.,  0. ]
                 ,[0. , 0. , 0. ,1. , 0.,  0. ]
                 ,[0. , 0., 0. , 0.  , 1., 0. ]
                 ,[0. , 0., 0. , 0.  , 0,  0 ]])
e = torch.tensor(ee)
F = torch.tensor(FF)
BT = torch.tensor(BBT)
BS = torch.tensor(BBS)
eps = 1e-20
dim = 3
alpha = 0.01

class RangingSensors3D(object):
    

    def __init__(self,states):
        self.states = states #sensor states
        
    def transition(x,u):
        '''state transition of sensors'''
        return FF@x+BBS@u
    
    def transition_target(x,a):
        return FF@x+BBT@a
        
    def hx(self,x):
        '''mapping states to observations num_targets by num_sensors'''
        v = (np.expand_dims(x,2)-np.expand_dims(self.states,1)).swapaxes(0,1)
        return 2*ci*norm(ee@(v+eps),axis=1)
    
    def H(self,x):
        '''Jocobian of hx at x'''
        v = (np.expand_dims(x,2)-np.expand_dims(self.states,1)).swapaxes(0,1)
        denom = norm(ee@(v+eps),axis=1,keepdims=True)
        HJ = 2*ci*ee@v/denom
        return HJ
    
    def sx(self,x):
        '''mapping states to noise'''
        n_sensors = self.states.shape[1]
        n_targets = x.shape[1]
        s = zeros((n_targets,n_sensors,n_sensors))
        d = self.hx(x)
        for i in range(n_targets):
            s[i] = diag(1+alpha*d[i])
        s = s*(ci**2)
        return s
    
    def S(self,x):
        '''Jocobian of sx at x'''
        n_targets = x.shape[1]
        n_sensors = self.states.shape[1]
        dcov = zeros((n_targets,dim,n_sensors,n_sensors))
        dmu = self.H(x)
        for i in range(n_targets):
            for j in range(dim):
                dcov[i,j] = diag(dmu[i,j])
        dcov *= (alpha*(ci**2))
        return dcov
    
class RangeSensor3DTorchUtils: 
    

  
    def rescale(self,cost):
        return cost
    
    def transition(self,x,u):
        '''state transition of sensors'''
        return F@x+BS@u
    
    def transition_target(self,x,a):
        return F@x+BT@a
    
    def transition_matrix(self,P,h,l):
        return F@P@F.T+BT@torch.diag((h-l)**2)@BT.T/12
    
    def hx(self,x,xs):
        '''mapping states to observations num_targets by num_sensors'''
        v = (torch.unsqueeze(x,2)-torch.unsqueeze(xs,1)).swapaxes(0,1)
        return 2*ci*torch.linalg.norm(e@(v+eps),axis=1)
    
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
        denom = torch.linalg.norm(e@(v+eps),axis=1,keepdims=True)
        HJ = 2*ci*e@v/denom
        return HJ
    
    def S(self,x,xs):
        '''Jocobian of sx at x'''
        n_targets = x.shape[1]
        n_sensors = xs.shape[1]
        dcov = torch.zeros((n_targets,dim,n_sensors,n_sensors))
        dmu = self.H(x,xs)
        for i in range(n_targets):
            for j in range(dim):
                dcov[i,j] = torch.diag(dmu[i,j])
        dcov *= (alpha*(ci**2))
        return dcov
    
    def estimate_velocity(self,x,x_prev):
        return x[dim:,:]-x_prev[dim:,:]
    
    
    def abs_err(self,x,est):
        return norm(ee@(x-est),axis=0)
    
    def get_F(self):
        return F
    
__all__=['RangingSensors3D','RangeSensor3DTorchUtils']