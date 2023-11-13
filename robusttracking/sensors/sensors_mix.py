# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, unicode_literals)

import numpy as np
from numpy.linalg import norm,inv
from numpy import array,trace,diag,zeros,ones,full,vstack,hstack

from .sensors_doppler import *
from .sensors_ranging import *
from .sensors import *

import torch

class MixedRangingDoppler3DFull(Sensor):
    
    def __init__(self,n_rangingsensor,n_static_ranging=0,n_static_doppler=0,doppler_noise=1):
        self.n_rangingsensor = n_rangingsensor
        self.n_static_ranging = n_static_ranging
        self.n_static_doppler = n_static_doppler
        
        self.ranging = RangingSensor3DFull(n_static=n_static_ranging)
        self.doppler = DopplerSensorFull(n_static=n_static_doppler,noise=doppler_noise)

        
    def transition(self,x,u):
        '''state transition of sensors'''
        x_ = zeros(x.shape)
        x_[:,:self.n_rangingsensor] = self.ranging.transition(x[:,:self.n_rangingsensor],u[:,:(self.n_rangingsensor-self.n_static_ranging)])
        x_[:,self.n_rangingsensor:] = self.doppler.transition(x[:,self.n_rangingsensor:],u[:,(self.n_rangingsensor-self.n_static_ranging):])
        return x_
        
    def hx(self,x,xs):
        '''mapping states to observations'''
        h1 = self.ranging.hx(x,xs[:,:self.n_rangingsensor])
        h2 = self.doppler.hx(x,xs[:,self.n_rangingsensor:])
        return hstack([h1,h2])
    
    def H(self,x,xs):
        '''Jocobian of hx at x'''
        H1 = self.ranging.H(x,xs[:,:self.n_rangingsensor])
        H2 = self.doppler.H(x,xs[:,self.n_rangingsensor:])
        n_targets,dim_1,n_sensors_1 = H1.shape
        _,dim_2,n_sensors_2 = H2.shape
        n_sensors = n_sensors_1 + n_sensors_2
        HH = zeros((n_targets,dim_2,n_sensors))
        HH[:,:dim_1,:n_sensors_1] = H1
        HH[:,:dim_2,n_sensors_1:] = H2
        return HH
    
    def sx(self,x,xs):
        '''mapping states to noise'''
        s1 = self.ranging.sx(x,xs[:,:self.n_rangingsensor])
        s2 = self.doppler.sx(x,xs[:,self.n_rangingsensor:])
        
        dim,n_targets = x.shape
        n_sensors_1 = s1.shape[-1]
        n_sensors_2 = s2.shape[-1]
        n_sensors = n_sensors_1 + n_sensors_2
        
        se = zeros((n_targets,n_sensors,n_sensors))
        se[:,:n_sensors_1,:n_sensors_1]  = s1
        se[:,n_sensors_1:,n_sensors_1:]  = s2
    
        return se
    
    def S(self,x,xs):
        '''Jocobian of sx at x'''
        S1 = self.ranging.S(x,xs[:,:self.n_rangingsensor]) 
        S2 = self.doppler.S(x,xs[:,self.n_rangingsensor:]) 
        n_targets,dim_1,n_sensors_1,_ = S1.shape
        _,dim_2,n_sensors_2,_ = S2.shape
        n_sensors = n_sensors_1+n_sensors_2
        dcov = zeros((n_targets,dim_2,n_sensors,n_sensors))
        dcov[:,:dim_1,:n_sensors_1,:n_sensors_1] = S1
        dcov[:,:dim_2,n_sensors_1:,n_sensors_1:] = S2
        return dcov
    
class MixedRangingDoppler3DFullT(Sensor): 
    
    def __init__(self,n_rangingsensor,n_static_ranging,n_static_doppler,doppler_noise=1):
        self.n_rangingsensor = n_rangingsensor
        self.n_static_ranging = n_static_ranging
        self.n_static_doppler = n_static_doppler
        
        self.ranging = RangingSensor3DFullT(n_static=n_static_ranging)
        self.doppler = DopplerSensorFullT(n_static=n_static_doppler,noise=doppler_noise)
    
    def transition(self,x,u):
        '''state transition of sensors'''
        x_ = torch.zeros(x.shape)
        x_[:,:self.n_rangingsensor] = self.ranging.transition(x[:,:self.n_rangingsensor],u[:,:(self.n_rangingsensor-self.n_static_ranging)])
        x_[:,self.n_rangingsensor:] = self.doppler.transition(x[:,self.n_rangingsensor:],u[:,(self.n_rangingsensor-self.n_static_ranging):])
        return x_
    
    def hx(self,x,xs):
        '''mapping states to observations'''
        h1 = self.ranging.hx(x,xs[:,:self.n_rangingsensor])
        h2 = self.doppler.hx(x,xs[:,self.n_rangingsensor:])
        return torch.hstack([h1,h2])

    def sx(self,x,xs):
        '''mapping states to noise'''
        s1 = self.ranging.sx(x,xs[:,:self.n_rangingsensor])
        s2 = self.doppler.sx(x,xs[:,self.n_rangingsensor:])
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
        H1 = self.ranging.H(x,xs[:,:self.n_rangingsensor])
        H2 = self.doppler.H(x,xs[:,self.n_rangingsensor:])
        n_targets,dim_1,n_sensors_1 = H1.shape
        _,dim_2,n_sensors_2 = H2.shape
        n_sensors = n_sensors_1 + n_sensors_2
        HH = torch.zeros((n_targets,dim_2,n_sensors))
        HH[:,:dim_1,:n_sensors_1] = H1
        HH[:,:dim_2,n_sensors_1:] = H2
        return HH
    
    def S(self,x,xs):
        '''Jocobian of sx at x'''
        S1 = self.ranging.S(x,xs[:,:self.n_rangingsensor]) 
        S2 = self.doppler.S(x,xs[:,self.n_rangingsensor:]) 
        n_targets,dim_1,n_sensors_1,_ = S1.shape
        _,dim_2,n_sensors_2,_ = S2.shape
        n_sensors = n_sensors_1+n_sensors_2
        dcov = torch.zeros((n_targets,dim_2,n_sensors,n_sensors))
        dcov[:,:dim_1,:n_sensors_1,:n_sensors_1] = S1
        dcov[:,:dim_2,n_sensors_1:,n_sensors_1:] = S2
        return dcov
    
    