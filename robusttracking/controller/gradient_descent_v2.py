# -*- coding: utf-8 -*-
import numpy as np

import torch
from torch.linalg import inv
from torch import trace,zeros,eye,sigmoid,relu,dot,inverse
from torch.optim import Adam,AdamW


from .cost import crlb_torch,barrier_cost
from .sigmapoints import get_sigmoid_points
from ..sensors import is_ranging
torch.set_default_dtype(torch.double)

def reparam(u,lower,upper): #reparameteriz
    return sigmoid(u).mul(upper-lower)+lower

def forward(sensor_torch,x_s,x_t,u_s,rs,Fk,Bk,dim_x_s,W=None,rho=0,S=None,lb=None,ub=None,c_inv_hat = None):
    """ forward propagation of cost
    sensor_torch: sensor model
    x_s: d x M, state of sensors
    x_t: d x N, state of targets
    u_s: K x d x M, sensor control
    rs: bound of u_s 
    c_inv_hat: covariance of predicts of x_t at t+K steps
    w: weight of cost at each time step, w_k*g(x_s[k],x_t[k])
    """
    
    #time horizon K
    K = x_s.shape[0]//dim_x_s
    M = x_s.shape[1]
   
    #reparameterize
    if u_s.requires_grad:
        u_s_ = reparam(u_s,*rs)
    else:
        u_s_ = u_s
    
    x_s_ = Fk@x_s+Bk@u_s_
            
    
    #step cost
    mu = sensor_torch.hx(x_t,x_s_[-dim_x_s:,:])
    cov = sensor_torch.sx(x_t,x_s_[-dim_x_s:,:])
    d_mu = sensor_torch.H(x_t,x_s_[-dim_x_s:,:])
    d_cov = sensor_torch.S(x_t,x_s_[-dim_x_s:,:])
    l = crlb_torch(mu,cov,d_mu,d_cov,W,c_inv_hat)
    
    if rho>0:
        return l + rho*barrier_cost(x_s_,S,lb,ub)/K/M
    else:
        return l

def step(sensor_torch,opts,x_s,x_t,u_s,rs,Fk,Bk,dim_x_s,W,rho=0,S=None,lb=None,ub=None,c_inv_hat = None):
    """ One step of gradient method
    x_s: d x M, state of sensors
    x_t: d x N, state of targets
    u_s: K x d x M, sensor control
    v: d x N, target uncertainty
    rs: bound of u_s 
    """
        
    for opt in opts:
        opt.zero_grad()
   
    loss = forward(sensor_torch,x_s,x_t,u_s,rs,Fk,Bk,dim_x_s,W,rho=rho,S=S,lb=lb,ub=ub,c_inv_hat = c_inv_hat)
        
    loss.backward()
    
    for opt in opts:
        opt.step()
    
    with torch.no_grad():
        loss_val = loss.item()
            
    return loss_val



def create_sigmapoints(K,target_torch,x_t,Q,C_prev,alpha=0.95):
    dimx,n_target = x_t.shape
    dim = dimx//2
    
    # eta_0 = [x_t_0,a_1,a_2,...a_{k-1}]
    eta_0 = torch.nn.functional.pad(x_t,(0,0,0,dim*(K)))
  
    x_t_set = []
    c_inv_set = []
    for it in range(n_target):
        #1. generate sigma points
        
        C0 = torch.block_diag(*([C_prev[it]]+K*[Q[dim:,dim:]]))
        
        eta_sets,wmx0,wcx0,wcxi = get_sigmoid_points(eta_0[:,it].numpy(),C0.numpy(),dimx, alpha)
        eta_sets = torch.tensor(eta_sets)
        #2. predict
        _,N = eta_sets.shape
        x_pred = torch.zeros((dimx,N))
        weights_m = torch.zeros((N,1))
        weights_c = torch.zeros((N,1))
        for i in range(N):
            x_t_ = eta_sets[:dimx,[i]]
            for k in range(K):
                a_k = torch.zeros(x_t_.shape) 
                a_k[dim:] = eta_sets[dimx+dim*k:dimx+dim*(k+1),[i]]
                x_t_ = target_torch.transition(x_t_,a_k)
        
                
            x_pred[:,[i]] = x_t_.clone().detach()
            
            if i==0:
                weights_m[i] = wmx0
                weights_c[i] = wcx0
            else:
                weights_m[i] = wcxi
                weights_c[i] = wcxi
            
            x_t_set.append(x_t_[:,0].clone().detach())
            
        print(weights_m.sum(),weights_c.sum())
        x_pred0 = x_pred@weights_m
        C = (x_pred-x_pred0)@np.diag(weights_c[:,0])@(x_pred-x_pred0).T
        
        
        c_inv_set.append(torch.tile(inv(C).unsqueeze(0),(N,1,1)))
    return torch.vstack(x_t_set).T,torch.vstack(c_inv_set)

def construct_c_inv_hat(K,target_torch,Q,C_prev):
    n_target = C_prev.shape[0]
    C_hat = zeros(C_prev.shape)
    
    for it in range(n_target):
        C_ = C_prev[it].clone().detach()
        for k in range(K):
            C_ = target_torch.predictC(C_,Q)
        C_hat[it] = C_.clone().detach()
    return inverse(C_hat)

def move(K,sensor_torch,target_torch,x_s,x_t,lr,rs,n_dynamic_sensors,C_prev,take_sigmapoints=True,rho = 0,S = None,lb = None,ub = None, n_iter = 1000, Q = None, init_u_s = None,W = None):

    s_shape = x_s.shape
    t_shape = x_t.shape
    d = s_shape[0]//2

    if init_u_s is None:
        #init_u_s = 0.01*np.random.randn(K,s_shape[0],n_dynamic_sensors)
        #init_u_s[:,:d,:] = 0
        init_u_s = 0.01*np.random.randn(K*s_shape[0],n_dynamic_sensors)
    
    losses = []
    
    dim_x_s = d*2
    Fk = torch.zeros((K*dim_x_s,K*dim_x_s))
    Bk = torch.zeros((K*dim_x_s,K*dim_x_s))
    for i in range(K):
        Fi,Bi = sensor_torch.getFBk(i+1)
        Fk[i*dim_x_s:i*dim_x_s+dim_x_s,i*dim_x_s:i*dim_x_s+dim_x_s] = Fi
        Bk[i*dim_x_s:i*dim_x_s+dim_x_s,:i*dim_x_s+dim_x_s] = Bi

    u_s = torch.tensor(init_u_s, requires_grad = True)
    
    x_s_ = torch.tile(x_s,(K,1))
    
    S = torch.block_diag(*[S for i in range(K)])
    lb = torch.tile(lb,(K,1))
    ub = torch.tile(ub,(K,1))
    opts = [Adam([u_s],lr = lr)]
    
    if take_sigmapoints:
        x_t_,c_inv_hat = create_sigmapoints(K,target_torch,x_t,Q,C_prev)
    else:
        x_t_ = x_t
        c_inv_hat =  construct_c_inv_hat(K,target_torch,Q,C_prev)
    
    print('take_sigmapoints',take_sigmapoints,'target shape',x_t_.shape,'c_inv_hat shape',c_inv_hat.shape,"?")
    
    #test
    #from torch.autograd import gradcheck,gradgradcheck
    # Perform backward pass
    #loss_test = lambda u : forward(sensor_torch,x_s_,x_t_,u,rs,Fk,Bk,dim_x_s,None,rho=0,S=S,lb=lb,ub=ub,c_inv_hat = c_inv_hat)
    #print(gradcheck(loss_test, (u_s,),  eps=1e-8))
    #print(gradgradcheck(loss_test, (u_s,), eps=1e-8))
    
    
    
    best_u = None
    best_l = np.inf
    for i in range (n_iter):
        loss_val = step(sensor_torch,opts,x_s_,x_t_,u_s,rs,Fk,Bk,dim_x_s,W=W,rho=rho,S=S,lb=lb,ub=ub,c_inv_hat = c_inv_hat)
        losses.append(loss_val)
        
        if loss_val==np.inf:
            break
        elif loss_val<best_l:
            best_l = loss_val
            best_u = reparam(u_s.clone().detach(),*rs)
     
    
    return best_u,losses