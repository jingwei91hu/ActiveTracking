# -*- coding: utf-8 -*-
from abc import ABC,abstractmethod
import torch
from numpy import array,zeros,block,eye

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

class Linear3DFullActor(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,F3D,B3D_Full)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,F3D,B3D_Full)
        
    def getUniformQ(self,umax):
        return self._getUniformQ(Q0_3D.copy(),3,umax)
    
class Linear3D2DActor(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,F3D,B3D_2D)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,F3D,B3D_2D)
    
    def getUniformQ(self,umax):
        return self._getUniformQ(Q0_3D.copy(),3,umax)
    
class Linear3DFullActorT(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,TF3D,TB3D_Full)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,TF3D,TB3D_Full)
    
    def getUniformQ(self,umax):
        return self._getUniformQ(TQ0_3D.clone().detach(),3,umax)
    
class Linear3D2DActorT(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,TF3D,TB3D_2D)

    def predictC(self,C,Q):
        return self._predictC(C,Q,TF3D,TB3D_2D)
    
    def getUniformQ(self,umax):
        return self._getUniformQ(TQ0_3D.clone().detach(),3,umax)
    
class Linear2DActor(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,F2D,B2D)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,F2D,B2D)
    
    def getUniformQ(self,umax):
        return self._getUniformQ(Q0_2D.copy(),2,umax)
    
class Linear2DActorT(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,TF2D,TB2D)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,TF2D,TB2D)
    
    def getUniformQ(self,umax):
        return self._getUniformQ(TQ0_2D.clone().detach(),2,umax)
    