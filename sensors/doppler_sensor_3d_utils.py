# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, unicode_literals)

import numpy as np
from numpy.linalg import norm,inv
from numpy import array,trace,diag,zeros,ones,full

import torch


torch.set_default_tensor_type(torch.DoubleTensor)
ci = (1e-8)/3
fc = 2.3*1e9
eev = array([[0.,0.,0.,1.,0.,0.]
            ,[0.,0.,0.,0.,1.,0.]
            ,[0.,0.,0.,0.,0.,1.]])

eep = array([[1.,0.,0.,0.,0.,0.]
                ,[0.,1.,0.,0.,0.,0.]
                ,[0.,0.,1.,0.,0.,0.]])

epep = array([[0.,0.,0.,1.,0.,0.]
                ,[0.,0.,0.,0.,1.,0.]
                ,[0.,0.,0.,0.,0.,1.]
                ,[1.,0.,0.,0.,0.,0.]
                ,[0.,1.,0.,0.,0.,0.]
                ,[0.,0.,1.,0.,0.,0.]])

eepp = array([[1.,0.,0.,0.,0.,0.]
                ,[0.,1.,0.,0.,0.,0.]
                ,[0.,0.,1.,0.,0.,0.]
                ,[0.,0.,0.,0.,0.,0.]
                ,[0.,0.,0.,0.,0.,0.]
                ,[0.,0.,0.,0.,0.,0.]])
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

ep = torch.tensor(eep)
pep = torch.tensor(epep)
epp = torch.tensor(eepp)

ev = torch.tensor(eev)
F = torch.tensor(FF)
BT = torch.tensor(BBT)
BS = torch.tensor(BBS)
noise = 100



class DopplerSensors3D(object):
    
    def __init__(self,states):
        self.states = states #sensor states
        self.sigma = zeros(1)
        self.s_deri = zeros(1)

    def transition(x,u):
        '''state transition of sensors'''
        return FF@x+BBS@u
    
    def transition_target(x,a):
        return FF@x+BBT@a
        
    def hx(self,x):
        '''mapping states to observations'''
        v = (np.expand_dims(x,2)-np.expand_dims(self.states,1)).swapaxes(0,1)
        d = norm(eep@v,axis=1) + (1e-20)
        return -fc*ci*np.einsum('ijk,ijk->ik',eev@v,eep@v)/d
    
    def H(self,x):
        '''Jocobian of hx at x'''
        v = (np.expand_dims(x,2)-np.expand_dims(self.states,1)).swapaxes(0,1)
        d = norm(eep@v,axis=1,keepdims=True) + (1e-20)
        s = np.expand_dims(np.einsum('ijk,ijk->ik',eev@v,eep@v),1)
        HM = (-fc*ci*(epep@v/d -  eepp@v/(d**3) * s))
        return HM
    
    def sx(self,x):
        '''mapping states to noise'''
        n_targets = x.shape[-1]
        n_sensors = self.states.shape[-1]

        if self.sigma.shape!=(n_targets,n_sensors,n_sensors):
            self.sigma = zeros((n_targets,n_sensors,n_sensors))
            for i in range(n_targets):
                self.sigma[i] = diag(ones(n_sensors))
            self.sigma *= noise
        return self.sigma
    
    def S(self,x):
        '''Jocobian of sx at x'''
        dim,n_targets = x.shape
        n_sensors = self.states.shape[-1]
        if self.s_deri.shape!=(n_targets,dim,n_sensors,n_sensors):
            self.s_deri = zeros((n_targets,dim,n_sensors,n_sensors))
        return self.s_deri
    
class DopplerSensor3DTorchUtils: 

    
    def __init__(self):
        self.sigma=torch.zeros(1)
        self.s_deri = torch.zeros(1)
    
    def rescale(self,cost):
        return cost
    
    def transition(self,x,u):
        '''state transition of sensors'''
        return F@x+BS@u
    
    def transition_target(self,x,a):
        return F@x+BT@a
    
    def transition_matrix(self,P):
        return F@P@F.T
    
    def hx(self,x,xs):
        '''mapping states to observations'''
        v = (torch.unsqueeze(x,2)-torch.unsqueeze(xs,1)).swapaxes(0,1)
        d = torch.linalg.norm(ep@v,axis=1) + (1e-20)
        return -fc*ci*torch.einsum('ijk,ijk->ik',ev@v,ep@v)/d
    
    def H(self,x,xs):
        '''Jocobian of hx at x'''
        v = (torch.unsqueeze(x,2)-torch.unsqueeze(xs,1)).swapaxes(0,1)
        d = torch.linalg.norm(ep@v,axis=1,keepdims=True) + (1e-20)
        s = torch.einsum('ijk,ijk->ik',ev@v,ep@v).unsqueeze(1)
        HM = (-fc*ci*(pep@v/d -  epp@v/(d**3) * s))
        return HM
    
    def sx(self,x,xs):
        '''mapping states to noise'''
        n_targets = x.shape[-1]
        n_sensors = xs.shape[-1]

        if self.sigma.shape!=(n_targets,n_sensors,n_sensors):
            self.sigma = torch.zeros((n_targets,n_sensors,n_sensors))
            for i in range(n_targets):
                self.sigma[i] = torch.diag(torch.ones(n_sensors))
            self.sigma *= noise
        return self.sigma
    
    def S(self,x,xs):
        dim,n_targets = x.shape
        n_sensors = xs.shape[-1]
        if self.s_deri.shape!=(n_targets,dim,n_sensors,n_sensors):
            self.s_deri = torch.zeros((n_targets,dim,n_sensors,n_sensors))
        return self.s_deri
    

    def abs_err(self,x,est):
        return norm(eev@(x-est),axis=0)
    
    def get_F(self):
        return F
    
__all__=['DopplerSensors3D','DopplerSensor3DTorchUtils']