# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, unicode_literals)

import numpy as np
from numpy.linalg import norm,inv
from numpy import array,trace,diag,zeros,ones,full,vstack,hstack
from . doppler_sensor_3d_utils_all import *
from . range_sensor_3d_utils_all import *

import torch

class EnsembledSensors3D_2(object):
    
    def __init__(self,n_rangesensor,states):
        self.states = states #sensor states
        self.n_rangesensor = n_rangesensor
        self.ranging = RangingSensors3D_2(states[:,:n_rangesensor])
        self.doppler = DopplerSensors3D_2(states[:,n_rangesensor:])
        
    def sync_states(self):
        self.ranging.states = self.states[:,:self.n_rangesensor]
        self.doppler.states = self.states[:,self.n_rangesensor:]
        
    def transition(x,u):
        '''state transition of sensors'''
        return self.ranging.transition(x,u)
    
    def transition_target(x,a):
        return self.ranging.transition_target(x,a)
        
    def hx(self,x):
        '''mapping states to observations'''
        self.sync_states()
        h1 = self.ranging.hx(x)
        h2 = self.doppler.hx(x)
        return hstack([h1,h2])
    
    def H(self,x):
        '''Jocobian of hx at x'''
        self.sync_states()
        H1 = self.ranging.H(x)
        H2 = self.doppler.H(x)
        n_targets,dim_1,n_sensors_1 = H1.shape
        _,dim_2,n_sensors_2 = H2.shape
        n_sensors = n_sensors_1 + n_sensors_2
        HH = zeros((n_targets,dim_2,n_sensors))
        HH[:,:dim_1,:n_sensors_1] = H1
        HH[:,:dim_2,n_sensors_1:] = H2
        return HH
    
    def sx(self,x):
        '''mapping states to noise'''
        self.sync_states()
        s1 = self.ranging.sx(x)
        s2 = self.doppler.sx(x)
        
        dim,n_targets = x.shape
        n_sensors_1 = s1.shape[-1]
        n_sensors_2 = s2.shape[-1]
        n_sensors = n_sensors_1 + n_sensors_2
        
        se = zeros((n_targets,n_sensors,n_sensors))
        se[:,:n_sensors_1,:n_sensors_1]  = s1
        se[:,n_sensors_1:,n_sensors_1:]  = s2
    
        return se
    
    def S(self,x):
        '''Jocobian of sx at x'''
        self.sync_states()
        
        S1 = self.ranging.S(x) 
        S2 = self.doppler.S(x) 
        n_targets,dim_1,n_sensors_1,_ = S1.shape
        _,dim_2,n_sensors_2,_ = S2.shape
        n_sensors = n_sensors_1+n_sensors_2
        dcov = zeros((n_targets,dim_2,n_sensors,n_sensors))
        dcov[:,:dim_1,:n_sensors_1,:n_sensors_1] = S1
        dcov[:,:dim_2,n_sensors_1:,n_sensors_1:] = S2
        return dcov
    
class EnsembledSensor3DTorchUtils_2: 
    
    def __init__(self,n_rangesensor):
        self.n_rangesensor = n_rangesensor
        self.rs = RangeSensor3DTorchUtils_2()
        self.ds = DopplerSensor3DTorchUtils_2()
    
    def rescale(self,cost):
        return cost
    
    def transition(self,x,u):
        '''state transition of sensors'''
        return self.rs.transition(x,u)
    
    def transition_target(self,x,a):
        return self.rs.transition_target(x,a)
    
    def transition_matrix(self,P):
        return self.rs.transition_matrix(P)
        
    def hx(self,x,xs):
        '''mapping states to observations'''
        h1 = self.rs.hx(x,xs[:,:self.n_rangesensor])
        h2 = self.ds.hx(x,xs[:,self.n_rangesensor:])
        return torch.hstack([h1,h2])

    def sx(self,x,xs):
        '''mapping states to noise'''
        s1 = self.rs.sx(x,xs[:,:self.n_rangesensor])
        s2 = self.ds.sx(x,xs[:,self.n_rangesensor:])
        dim,n_targets = x.shape
        n_sensors_1 = s1.shape[-1]
        n_sensors_2 = s2.shape[-1]
        n_sensors = n_sensors_1 + n_sensors_2
        
        se = torch.zeros((n_targets,n_sensors,n_sensors))
        
        se[:,:n_sensors_1,:n_sensors_1]  = s1
        se[:,n_sensors_1:,n_sensors_1:]  = s2
    
        return se
    
    def H(self,x,xs):
        '''Jocobian of hx at x'''
        H1 = self.rs.H(x,xs[:,:self.n_rangesensor])
        H2 = self.ds.H(x,xs[:,self.n_rangesensor:])
        n_targets,dim_1,n_sensors_1 = H1.shape
        _,dim_2,n_sensors_2 = H2.shape
        n_sensors = n_sensors_1 + n_sensors_2
        HH = torch.zeros((n_targets,dim_2,n_sensors))
        HH[:,:dim_1,:n_sensors_1] = H1
        HH[:,:dim_2,n_sensors_1:] = H2
        return HH
    
    def S(self,x,xs):
        '''Jocobian of sx at x'''
        S1 = self.rs.S(x,xs[:,:self.n_rangesensor]) 
        S2 = self.ds.S(x,xs[:,self.n_rangesensor:]) 
        n_targets,dim_1,n_sensors_1,_ = S1.shape
        _,dim_2,n_sensors_2,_ = S2.shape
        n_sensors = n_sensors_1+n_sensors_2
        dcov = torch.zeros((n_targets,dim_2,n_sensors,n_sensors))
        dcov[:,:dim_1,:n_sensors_1,:n_sensors_1] = S1
        dcov[:,:dim_2,n_sensors_1:,n_sensors_1:] = S2
        return dcov
    
    def abs_err(self,x,est):
        return self.rs.abs_err(x,est)
    
    def get_F(self):
        return F
    
    
__all__=['EnsembledSensors3D_2','EnsembledSensor3DTorchUtils_2']