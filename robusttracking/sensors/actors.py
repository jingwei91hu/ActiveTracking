# -*- coding: utf-8 -*-
from abc import ABC,abstractmethod
import torch
from numpy import array,zeros,block,eye
from numpy.linalg import matrix_power
import numpy as np
import jax.numpy as jnp
dt = 1

B3D_Full = block([[zeros((3,3)),0.5*(dt**2)*eye(3)],[zeros((3,3)),dt*eye(3)]])

B3D_2D = block([[zeros((3,3)),array([[0.5*(dt**2),0,0],[0,0.5*(dt**2),0],[0,0,0]])],[zeros((3,3)),dt*eye(3)]])

F2D = block([[eye(2),dt*eye(2)],[zeros((2,2)),eye(2)]])

B2D = block([[zeros((2,2)),0.5*(dt**2)*eye(2)],[zeros((2,2)),dt*eye(2)]])

F3D = block([[eye(3),dt*eye(3)],[zeros((3,3)),eye(3)]])

Q0_2D = zeros((4,4))
Q0_3D = zeros((6,6))

torch.set_default_tensor_type(torch.DoubleTensor)
#3-dimensional
TB3D_Full = torch.tensor(B3D_Full)
TB3D_2D = torch.tensor(B3D_2D)
TF3D = torch.tensor(F3D)
TQ0_3D = torch.tensor(Q0_3D)

#2-dimensional
TF2D = torch.tensor(F2D)
TB2D = torch.tensor(B2D)
TQ0_2D = torch.tensor(Q0_2D)

#jax
#2-dimensional
JF2D = jnp.array(F2D)
JB2D = jnp.array(B2D)
JQ0_2D = jnp.array(Q0_2D)
#3-dimensional
JF3D = jnp.array(F3D)
JB3D_Full = jnp.array(B3D_Full)
JQ0_3D = jnp.array(Q0_3D)

class Actor(ABC):
    def __init__(self,n_static,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_static = n_static
    
    def _transition(self,x,u,F,B):
        x_ = F@x
        x_[:,self.n_static:] += B@u
        return x_
        
    def _predictC(self,C,Q,F,B):
        return F@C@F.T+B@Q@B.T
    
    def _getUniformQ(self,Q,d,umax):
        for i in range(d):
            Q[d+i,d+i] = ((2*umax)**2)/12
        return Q
    
    @abstractmethod
    def transition(self,x,u):
        pass
    
    @abstractmethod
    def predictC(self,C,Q):
        pass
    
    @abstractmethod
    def getUniformQ(self,umax):
        pass
    
    @abstractmethod
    def getFB(self):
        pass
    
    @abstractmethod
    def getFBk(self,K):
        pass
    
    def transitionk(self,x,u):
        K = u.shape[0]
        Fk,Bk = self.getFBk(K)
        return self._transition(x,u.reshape(Bk.shape[1],-1,order='C'),Fk,Bk)
    
class Linear3DFullActor(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,F3D,B3D_Full)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,F3D,B3D_Full)
        
    def getUniformQ(self,umax):
        return self._getUniformQ(Q0_3D.copy(),3,umax)
    
    def getFB(self):
        return F3D,B3D_Full
    
    def getFBk(self,K):
        Fk = np.linalg.matrix_power(F3D,K)
        Bk = np.hstack([np.linalg.matrix_power(F3D,K-i-1)@B3D_Full for i in range(K)])
        return Fk,Bk

class Linear3DFullActorJ(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
    def transition(self,x,u):
        x_ = JF3D@x
        x_ = x_.at[:,self.n_static:].set(x_[:,self.n_static:]+JB3D_Full@u)
        return x_
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,JF3D,JB3D_Full)
        
    def getUniformQ(self,umax):
        return self._getUniformQ(JQ0_3D.copy(),3,umax)
    
    def getFB(self):
        return F3D,B3D_Full
    
    def getFBk(self,K):
        Fk = jnp.linalg.matrix_power(JF3D,K)
        Bk = jnp.hstack([jnp.linalg.matrix_power(JF3D,K-i-1)@JB3D_Full for i in range(K)])
        return Fk,Bk
    
class Linear3D2DActor(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,F3D,B3D_2D)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,F3D,B3D_2D)
    
    def getUniformQ(self,umax):
        return self._getUniformQ(Q0_3D.copy(),3,umax)
    
    def getFB(self):
        return F3D,B3D_2D
    
    def getFBk(self,K):
        Fk = matrix_power(F3D,K)
        Bk = np.hstack([np.linalg.matrix_power(F3D,K-i-1)@B3D_2D for i in range(K)])
        return Fk,Bk
    

    
class Linear3DFullActorT(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,TF3D,TB3D_Full)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,TF3D,TB3D_Full)
    
    def getUniformQ(self,umax):
        return self._getUniformQ(TQ0_3D.clone().detach(),3,umax)
    
    def getFB(self):
        return TF3D,TB3D_Full
    
    def getFBk(self,K):
        Fk = torch.linalg.matrix_power(TF3D,K)
        Bk = torch.hstack([torch.linalg.matrix_power(TF3D,K-i-1)@TB3D_Full for i in range(K)])
        return Fk,Bk
    
class Linear3D2DActorT(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,TF3D,TB3D_2D)

    def predictC(self,C,Q):
        return self._predictC(C,Q,TF3D,TB3D_2D)
    
    def getUniformQ(self,umax):
        return self._getUniformQ(TQ0_3D.clone().detach(),3,umax)
    
    def getFB(self):
        return TF3D,TB3D_2D
    
    def getFBk(self,K):
        Fk = torch.linalg.matrix_power(TF3D,K)
        Bk = torch.hstack([torch.linalg.matrix_power(TF3D,K-i-1)@TB3D_2D for i in range(K)])
        return Fk,Bk
    
class Linear2DActor(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,F2D,B2D)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,F2D,B2D)
    
    def getUniformQ(self,umax):
        return self._getUniformQ(Q0_2D.copy(),2,umax)
    
    def getFB(self):
        return F2D,B2D
    
    def getFBk(self,K):
        Fk = matrix_power(F2D,K)
        Bk = np.hstack([matrix_power(F2D,K-i-1)@B2D for i in range(K)])
        return Fk,Bk
    
class Linear2DActorJ(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        x_ = JF2D@x
        x_ = x_.at[:,self.n_static:].set(x_[:,self.n_static:]+JB2D@u)
        return x_
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,JF2D,JB2D)
    
    def getUniformQ(self,umax):
        return self._getUniformQ(JQ0_2D.copy(),2,umax)
    
    def getFB(self):
        return JF2D,JB2D
    
    def getFBk(self,K):
        Fk = jnp.linalg.matrix_power(JF2D,K)
        Bk = jnp.hstack([jnp.linalg.matrix_power(JF2D,K-i-1)@JB2D for i in range(K)])
        return Fk,Bk
    
class Linear2DActorT(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,TF2D,TB2D)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,TF2D,TB2D)
    
    def getUniformQ(self,umax):
        return self._getUniformQ(TQ0_2D.clone().detach(),2,umax)
    
    def getFB(self):
        return TF2D,TB2D
    
    def getFBk(self,K):
        Fk = torch.linalg.matrix_power(TF2D,K)
        Bk = torch.hstack([torch.linalg.matrix_power(TF2D,K-i-1)@TB2D for i in range(K)])
        return Fk,Bk