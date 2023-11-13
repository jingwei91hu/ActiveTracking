# -*- coding: utf-8 -*-
import numpy as np

import torch
from torch.linalg import inv
from torch import trace,zeros,eye,sigmoid,dot
from torch.optim import Adam


from .cost import crlb_torch

def proj(u,r):
    return u*r/torch.clamp(torch.linalg.norm(u,axis=1,keepdim=True)**2,min=r)

def proj_v(v,r):
    return v*r/torch.clamp(torch.linalg.norm(v,axis=0,keepdim=True)**2,min=r)

def forward(sensor_torch,target_torch,x_s,x_t,u_s,u_t,v,rs,rt,rv,C_sqt,w,gamma = 1):
    """ forward propagation of cost
    sensor_torch: simulation of sensors 
    target_torch: simulation of target
    x_s: d x M, state of sensors
    x_t: d x N, state of targets
    u_s: K x d x M, sensor control
    u_t: K x d x N, target control
    v: d x N, target uncertainty
    rs: bound of u_s 
    rt: bound of u_t 
    rv: bound of v 
    C_sqt: ellipsoid transition of v
    w: weight of cost at each time step, w_k*g(x_s[k],x_t[k]) 
    gamma: weight of velocity cost
    """
    
    #time horizon K
    K = u_t.shape[0]
    if u_s.requires_grad:
        u_s = proj(u_s,rs)
    
    if u_t.requires_grad:
        u_t = proj(u_t,rt)
        
        
        
    #reparameterize x_0
    if v.requires_grad:
        v_ = zeros(v.shape)
        for n_target in range(v.shape[1]):
            v_[:,n_target] =  C_sqt[n_target,:,:]@v[:,n_target]      
        v = v_
        v = proj_v(v,rv)
    
    #initialize variables
    L = zeros(K)
    x_s_ = zeros((K,*x_s.shape))
    x_t_ = zeros((K,*x_t.shape))
    
    for k in range(K):
        if k == 0:
            x_s_[k] = sensor_torch.transition(x_s,u_s[k])
            x_t_[k] = target_torch.transition(x_t+v,u_t[k])
        else:
            x_s_[k] = sensor_torch.transition(x_s_[k-1],u_s[k])
            x_t_[k] = target_torch.transition(x_t_[k-1],u_t[k])
        
        #step cost
        mu = sensor_torch.hx(x_t_[k],x_s_[k])
        cov = sensor_torch.sx(x_t_[k],x_s_[k])
        d_mu = sensor_torch.H(x_t_[k],x_s_[k])
        d_cov = sensor_torch.S(x_t_[k],x_s_[k])
        L[k] = crlb_torch(mu,cov,d_mu,d_cov,gamma)
    
    return dot(L,w)



def step_proj(sensor_torch,target_torch,opts,x_s,x_t,u_s,u_t,v,rs,rt,rv,C_sqt,w,ekf=False,gamma = 1):
    """ One step of gradient method
    x_s: d x M, state of sensors
    x_t: d x N, state of targets
    u_s: K x d x M, sensor control
    u_t: K x d x N, target control
    v: d x N, target uncertainty
    rs: bound of u_s 
    rt: bound of u_t 
    rv: bound of v 
    C_sqt: ellipsoid transition of v
    w: weight of cost at each time step, w_k*g(x_s[k],x_t[k]) 
    gamma: weight of velocity cost
    """
    for opt in opts:
        opt.zero_grad()
   
    if ekf:
        loss = forward_ekf(sensor_torch,target_torch,x_s,x_t,u_s,u_t,v,rs,rt,rv,C_sqt,w)
    else:
        loss = forward(sensor_torch,target_torch,x_s,x_t,u_s,u_t,v,rs,rt,rv,C_sqt,w,gamma=gamma)
        
    loss.backward()
    
    for opt in opts:
        opt.step()
    
    with torch.no_grad():
        loss_val = loss.item()
        
 
    return loss_val


def move_proj(K,sensor_torch,target_torch,x_s,x_t,lr,rs,rt,C_sqt,w,n_dynamic_sensors,is_target=True,n_iter = 1000, ekf = False,gamma = 1):
    """
     if is_target = True, rt works, otherwise rs works. 
     n_static only works when is_target=False
    """
    s_shape = x_s.shape
    t_shape = x_t.shape
    
    d = s_shape[1]//2
    
    if is_target:
        u_s = torch.zeros((K,*s_shape))
        init_u_t = 0.01*np.random.randn(K,*t_shape)
        init_u_t[:,:d,:] = 0
        u_t = torch.tensor(init_u_t, requires_grad = True)
    else:
        init_u_s = 0.01*np.random.randn(K,s_shape[0],n_dynamic_sensors)
        init_u_s[:,:d,:] = 0
        u_s = torch.tensor(init_u_s, requires_grad = True)
        u_t = torch.zeros((K,*t_shape))
        
    v = torch.zeros(t_shape)
    rv = torch.zeros(2)
    
    losses = []
    
    opts = [Adam([u_t if is_target else u_s],lr = lr)]
    
    if is_target:
        opts[0].param_groups[0]['lr'] *= -1 #change descending to ascending
    
    for i in range (n_iter):
        loss_val = step_proj(sensor_torch,target_torch,opts,x_s,x_t,u_s,u_t,v,rs,rt,rv,C_sqt,w,ekf = ekf,gamma = gamma)
        losses.append(loss_val)
        

        if loss_val==np.inf:
            break
           
    if is_target:
        u = proj(u_t.detach(),rt)
    else:
        u = proj(u_s.detach(),rs)
        
    return u,losses

def robustmove_proj(K,sensor_torch,target_torch,x_s,x_t,lr_s,lr_t,rs,rt,rv,C_sqt,w,n_dynamic_sensors,n_iter = 1000,ekf=False,gamma = 1):
    s_shape = x_s.shape
    t_shape = x_t.shape
    d = s_shape[1]//2
    
    init_u_s = 0.01*np.random.randn(K,s_shape[0],n_dynamic_sensors)
    init_u_s[:,:d,:] = 0
    u_s = torch.tensor(init_u_s, requires_grad=True)
    
    init_u_t = 0.01*np.random.randn(K,*t_shape)
    init_u_t[:,:d,:] = 0
    u_t = torch.tensor(init_u_t, requires_grad=True)
    
    #uncertainty variable
    v = torch.zeros(t_shape,requires_grad=True)
    
    losses = []
    
    opt_t = Adam([u_t,v],lr = lr_t) 
    opt_t.param_groups[0]['lr'] *= -1
    
    opt_s = Adam([u_s],lr = lr_s)
    
    opts = [opt_s,opt_t]
    
    for i in range (n_iter):
        loss_val = step_proj(sensor_torch,target_torch,opts,x_s,x_t,u_s,u_t,v,rs,rt,rv,C_sqt,w,ekf=ekf,gamma=gamma)
        losses.append(loss_val)

    u_s_val = proj(u_s.detach(),rs)
    
    return u_s_val,losses
