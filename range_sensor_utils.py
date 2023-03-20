# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, unicode_literals)

import numpy as np
from numpy.linalg import norm,inv
from numpy import array,trace,diag,zeros,ones,full

import torch


ci = (10**-8)/3
ee = array([[1.,0.,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])
FF = array([[1., 0., 1., 0.],[0., 1., 0., 1.],[0., 0., 1., 0.],[0., 0., 0., 1.]])
BB = array([[0. , 0. , 0.5 , 0. ],[0. , 0. , 0. , 0.5 ],[0., 0. , 1. , 0. ],[0. , 0., 0. , 1. ]])


torch.set_default_tensor_type(torch.DoubleTensor)
e = torch.tensor(ee)
F = torch.tensor(FF)
B = torch.tensor(BB)

class RangingSensors(object):
    def __init__(self,states):
        self.states = states #sensor states
        
    def transition(x,u):
        '''state transition of sensors'''
        return FF@x+BB@u
        
    def hx(self,x):
        '''mapping states to observations'''
        return 2*ci*norm(ee.T@(x-self.states),axis=0)
    
    def H(self,x):
        '''Jocobian of hx at x'''
        denom = norm(ee.T@(x-self.states),axis=0)[None,:]
        HJ = 2*ci*ee.T@(x-self.states)/denom
        return HJ.T
    
    def sx(self,x):
        '''mapping states to noise'''
        #return diag(full(self.states.shape[1],ci**2))#
        return diag((ci**2)*(1+0.5*self.hx(x)))
    
    def S(self,x):
        '''Jocobian of sx at x'''
        #return zeros((*self.states.shape,self.states.shape[-1]))#
        dcov = zeros((*self.states.shape,self.states.shape[-1]))
        dmu = self.H(x)
        for m in range(self.states.shape[0]):
            dcov[m] = diag(dmu[:,m])
        dcov *= (0.05*(ci**2))
        return dcov
    
class RangeSensorTorchUtils: 
    
    @staticmethod
    def rescale(cost):
        return cost
    
    @staticmethod
    def transition(x,u):
        '''state transition of sensors'''
        return F@x+B@u
    @staticmethod
    def transition_matrix(P):
        return F@P@F.T
    
    @staticmethod    
    def hx(x,xs):
        '''mapping states to observations'''
        return 2*ci*torch.linalg.norm(e.T@(x-xs),axis=0)
    @staticmethod
    def sx(x,xs):
        '''mapping states to noise'''
        #return (ci**2)*torch.diag(torch.ones(xs.shape[-1]))
        mu = 2*ci*torch.linalg.norm(e.T@(x-xs),axis=0)
        return torch.diag((ci**2)*(1+0.5*mu))
    
    @staticmethod
    def H(x,xs):
        '''Jocobian of hx at x'''
        denom = torch.linalg.norm(e.T@(x-xs),axis=0)[None,:]
        HJ = 2*ci*e.T@(x-xs)/denom
        return HJ[:2,:].T
    @staticmethod
    def S(x,xs):
        '''Jocobian of sx at x'''
        #return torch.zeros((2,xs.shape[-1],xs.shape[-1]))#
        dcov =torch. zeros((2,xs.shape[-1],xs.shape[-1]))
        denom = torch.linalg.norm(e.T@(x-xs),axis=0)[None,:]
        HJ = 2*ci*e.T@(x-xs)/denom
        dmu = HJ[:2,:].T
        for m in range(2):
            dcov[m] = torch.diag(dmu[:,m])
        dcov *= (0.05*(ci**2))
        return dcov
    
    @staticmethod
    def estimate_velocity(x,x_prev):
        return x[2:,:]-x_prev[2:,:]
    
    @staticmethod
    def abs_err(x,est):
        return np.sqrt(((x[:2,:]-est[:2,:])**2).sum())