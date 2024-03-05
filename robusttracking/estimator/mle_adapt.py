import torch
from torch.linalg import inv,cholesky
from torch import trace,zeros,eye,sigmoid,dot
from torch.optim import Adam

#Separate Q for each target?

from torch.optim import Adam
from torch.linalg import inv
def forward(sensor_torch,target_torch,obs,x,x_pred,C_prev,xs,v,Q):
    mu = sensor_torch.hx(x,xs)
    cov = sensor_torch.sx(x,xs)
    v = torch.exp(v)
    
    muo = obs-mu
    muo = muo.unsqueeze(2)
    ll = (torch.bmm(torch.bmm(muo.mT,inv(cov)),muo).squeeze(-1).squeeze(-1)+torch.slogdet(cov)[1]).sum()
    
    
    
    C_pred = torch.zeros(C_prev.shape)
    mx = x.T-x_pred.T
    mx = mx.unsqueeze(2)
    n_targets = mu.shape[0]
    for i in range(n_targets):
        C_pred[i] = target_torch.predictC(C_prev[i],v[i]*Q)
        
    ll += (torch.bmm(torch.bmm(mx.mT,inv(C_pred)),mx).squeeze(-1).squeeze(-1)+torch.slogdet(C_pred)[1]).sum()
    return ll

def step(sensor_torch,target_torch,opt,obs,x,x_pred,C_prev,xs,v,Q):
    
    opt.zero_grad()
   
    loss = forward(sensor_torch,target_torch,obs,x,x_pred,C_prev,xs,v,Q)
        
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

def mle_adapt(sensor_torch,target_torch,x_pred,C_prev,obs,xs,Q,n_iter=3000):
    d,n_target = x_pred.shape
    x = x_pred.clone().detach().requires_grad_(True)
    v = torch.zeros(n_target).requires_grad_(True)
    losses = []
    
    opt = Adam([x,v],lr = 0.01) 
    for i in range (n_iter):
        loss_val = step(sensor_torch,target_torch,opt,obs,x,x_pred,C_prev,xs,v,Q)
        losses.append(loss_val)
    x_est = x.clone().detach()
    v_est = torch.exp(v.clone().detach())
    
    #Give error covariance
    F = fisher_information(sensor_torch.hx(x_est,xs),sensor_torch.sx(x_est,xs),sensor_torch.H(x_est,xs),sensor_torch.S(x_est,xs))
    df = F.shape[-1]
    C = torch.zeros((n_target,d,d))
    
    for j in range(n_target):
        f = inv(target_torch.predictC(C_prev[j],v_est[j]*Q))
        f[:df,:df] += F[j]
        C[j] = cholesky(inv(f)).T
    return x_est,C,v_est