import torch
from torch.linalg import inv,cholesky
from torch import trace,zeros,eye,sigmoid,dot
from torch.optim import Adam
            
import numpy as np

def forward(sensor_torch,obs,x,x_pred,xs,invA):
    z = proj(x,x_pred,invA)
    mu = sensor_torch.hx(z,xs)
    cov = sensor_torch.sx(z,xs)
    muo = obs-mu
    muo = muo.unsqueeze(2)
    ll = (torch.bmm(torch.bmm(muo.mT,inv(cov)),muo).squeeze(-1).squeeze(-1)+torch.slogdet(cov)[1]).sum()    
    return ll

def proj(x,x_pred,invA):
    x_ = x-x_pred
    z = torch.zeros(x.shape)
    for i in range(x.shape[-1]): 
        z[:,i] = x_[:,i]/torch.clamp(torch.pow(x_[:,i]@invA[i]@x_[:,i],0.5),min=1) + x_pred[:,i]
    return z

def step(sensor_torch,opt,invA,obs,x,x_pred,xs):
    
    opt.zero_grad()
   
    loss = forward(sensor_torch,obs,x,x_pred,xs,invA)
    loss.backward()
    opt.step()
    
    with torch.no_grad():
        loss_val = loss.item()
    return loss_val

def fisher_information(mu,cov,d_mu,d_cov):
    n_targets,dim,n_sensors = d_mu.shape
    I = zeros((n_targets,dim,dim))
    for k in range(n_targets):
        inv_cov = inv(cov[k])
        for i in range(dim):
            for j in range(dim):
                I[k,i,j] = d_mu[k,i,None,:]@inv_cov@d_mu[k,j,:,None] + 0.5*trace(inv_cov@d_cov[k,i,:,:]@inv_cov@d_cov[k,j,:,:])
    return I

def mle_constrained(sensor_torch,target_torch,x_pred,C_prev,obs,xs,Q,q_alpha,n_iter=2000,factor=0.9,tol=1e-6):
    x = (x_pred.clone().detach()+torch.randn(x_pred.shape)*1e-3).requires_grad_(True)
    
    invTau = torch.zeros(C_prev.shape)
    d,n_targets = x_pred.shape
    for i in range(n_targets):
        invTau[i] = inv(target_torch.predictC((1/(1-factor))*C_prev[i],(1/factor)*Q))
    
    losses = []
    
    opt = Adam([x],lr = 0.01) 
    
    for i in range (n_iter):
        loss_val = step(sensor_torch,opt,invTau,obs,x,x_pred,xs)
        losses.append(loss_val)
        if len(losses)>20 and np.std(losses[-20:-1])<tol:
            break
    
    with torch.no_grad():
        x_est = proj(x.detach(),x_pred,invTau)
    
    F = fisher_information(sensor_torch.hx(x_est,xs),sensor_torch.sx(x_est,xs),sensor_torch.H(x_est,xs),sensor_torch.S(x_est,xs))
    df = F.shape[-1]
    C_sqt = torch.zeros((n_targets,d,d))
    
    for i in range(n_targets):
        f = invTau[i]
        f = f*q_alpha
        f[:df,:df] += F[i]
        C_sqt[i] = cholesky(inv(f)*q_alpha).T
        
    return x_est,C_sqt,losses
