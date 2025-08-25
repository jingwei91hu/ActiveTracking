# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, unicode_literals)

import numpy as np
from numpy.linalg import norm,inv
from numpy import array,trace,diag,zeros,ones,full,vstack,hstack

from .sensors_doppler import *
from .sensors_ranging import *
from .sensors import *

import torch

class MixedRangingDoppler3DFull(Sensor,Linear3DFullActor):
    
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
    
class MixedRangingDoppler3DFullT(Sensor,Linear3DFullActorT): 
    
    def __init__(self,n_rangingsensor,n_static_ranging=0,n_static_doppler=0,doppler_noise=1):
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
    
    
class MixedRangingDoppler3DFullJ(Sensor,Linear3DFullActorJ):
    
    def __init__(self,n_rangingsensor,n_static_ranging=0,n_static_doppler=0,doppler_noise=1):
        self.n_static = n_static_ranging+n_static_doppler
        self.n_rangingsensor = n_rangingsensor
        self.n_static_ranging = n_static_ranging
        self.n_static_doppler = n_static_doppler
        
        self.ranging = RangingSensor3DFullJ(n_static=n_static_ranging)
        self.doppler = DopplerSensorFullJ(n_static=n_static_doppler,noise=doppler_noise)
    #Not Support static sensors anymore
        
    def transition(self,x,u):
        '''state transition of sensors'''
        X1 = self.ranging.transition(x[:,:self.n_rangingsensor],u[:,:(self.n_rangingsensor-self.n_static_ranging)])
        X2 = self.doppler.transition(x[:,self.n_rangingsensor:],u[:,(self.n_rangingsensor-self.n_static_ranging):])
        x_ = jnp.concatenate([X1,X2],axis=1)
        return x_
        
    def hx(self,x,xs):
        '''mapping states to observations'''
        h1 = self.ranging.hx(x,xs[:,:self.n_rangingsensor])
        h2 = self.doppler.hx(x,xs[:,self.n_rangingsensor:])
        return jnp.hstack([h1,h2])
    
    def H(self,x,xs):
        '''Jocobian of hx at x'''
        H1 = self.ranging.H(x,xs[:,:self.n_rangingsensor])
        H2 = self.doppler.H(x,xs[:,self.n_rangingsensor:])
        n_targets,dim_1,n_sensors_1 = H1.shape
        _,dim_2,n_sensors_2 = H2.shape
        n_sensors = n_sensors_1 + n_sensors_2
        
        padding = jnp.zeros((n_targets,dim_2-dim_1,n_sensors_1))
        HH = jnp.concatenate([jnp.concatenate([H1,padding],axis=1),H2],axis=2)
        #HH = zeros((n_targets,dim_2,n_sensors))
        #HH[:,:dim_1,:n_sensors_1] = H1
        #HH[:,:dim_2,n_sensors_1:] = H2
        #HH = jnp.array(HH)
                              
        return HH
    
    def sx(self,x,xs):
        '''mapping states to noise'''
        s1 = self.ranging.sx(x,xs[:,:self.n_rangingsensor])
        s2 = self.doppler.sx(x,xs[:,self.n_rangingsensor:])
        
        dim,n_targets = x.shape
        n_sensors_1 = s1.shape[-1]
        n_sensors_2 = s2.shape[-1]
        n_sensors = n_sensors_1 + n_sensors_2
        
        
        top = jnp.zeros((n_targets, n_sensors_1, n_sensors_2))  # shape (4, 5, 2)
        bottom = jnp.zeros((n_targets, n_sensors_2, n_sensors_1))  # shape (4, 5, 3)

        se = jnp.concatenate([jnp.concatenate([s1,top],axis=2),
        jnp.concatenate([bottom,s2],axis=2)],axis=1)

        #se = zeros((n_targets,n_sensors,n_sensors))
        #se[:,:n_sensors_1,:n_sensors_1]  = s1
        #se[:,n_sensors_1:,n_sensors_1:]  = s2
    
        #se = jnp.array(se)
        return se
    
    def S(self,x,xs):
        '''Jocobian of sx at x'''
        S1 = self.ranging.S(x,xs[:,:self.n_rangingsensor]) 
        S2 = self.doppler.S(x,xs[:,self.n_rangingsensor:]) 
        n_targets,dim_1,n_sensors_1,_ = S1.shape
        _,dim_2,n_sensors_2,_ = S2.shape
        n_sensors = n_sensors_1+n_sensors_2
        
        padding1 = jnp.zeros((n_targets,dim_2-dim_1,n_sensors_1,n_sensors_1))
        
        padding2 = jnp.zeros((n_targets,dim_2,n_sensors_2,n_sensors_1))
        
        padding3 = jnp.zeros((n_targets,dim_2,n_sensors_1,n_sensors_2))
        
        dcov = jnp.concatenate([jnp.concatenate([jnp.concatenate([S1,padding1],axis=1),padding2],axis=2),jnp.concatenate([padding3,S2],axis=2)],axis=3)
        #dcov = zeros((n_targets,dim_2,n_sensors,n_sensors))
        #dcov[:,:dim_1,:n_sensors_1,:n_sensors_1] = S1
        #dcov[:,:dim_2,n_sensors_1:,n_sensors_1:] = S2
        #dcov = jnp.array(dcov)
        return dcov