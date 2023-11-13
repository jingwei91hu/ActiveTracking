# -*- coding: utf-8 -*-
from abc import ABC,abstractmethod
import torch
from numpy import array

B3D_Full = array([[0. , 0. , 0., 0.5, 0., 0.]
                 ,[0. , 0. , 0. , 0., 0.5, 0.]
                 ,[0. , 0. , 0. , 0., 0.,  0.5 ]
                 ,[0. , 0. , 0. ,1. , 0.,  0. ]
                 ,[0. , 0., 0. , 0.  , 1., 0. ]
                 ,[0. , 0., 0. , 0.  , 0,  1. ]])

B3D_2D = array([[0. , 0. , 0., 0.5, 0., 0.]
                 ,[0. , 0. , 0. , 0., 0.5, 0.]
                 ,[0. , 0. , 0. , 0., 0.,  0. ]
                 ,[0. , 0. , 0. ,1. , 0.,  0. ]
                 ,[0. , 0., 0. , 0.  , 1., 0. ]
                 ,[0. , 0., 0. , 0.  , 0,  0 ]])

F2D = array([[1., 0., 1., 0.]
             ,[0., 1., 0., 1.]
             ,[0., 0., 1., 0.]
             ,[0., 0., 0., 1.]])

B2D = array([[0. , 0. , 0.5 , 0. ]
             ,[0. , 0. , 0. , 0.5 ]
             ,[0., 0. , 1. , 0. ]
             ,[0. , 0., 0. , 1. ]])

F3D = array([[1., 0., 0., 1., 0., 0.]
                 ,[0., 1., 0., 0., 1., 0.]
                 ,[0., 0., 1., 0., 0., 1.]
                 ,[0., 0., 0., 1., 0., 0.]
                 ,[0., 0., 0., 0., 1., 0.]
                 ,[0., 0., 0., 0., 0., 1.]])

torch.set_default_tensor_type(torch.DoubleTensor)
#3-dimensional
TB3D_Full = torch.tensor(B3D_Full)
TB3D_2D = torch.tensor(B3D_2D)
TF3D = torch.tensor(F3D)
#2-dimensional
TF2D = torch.tensor(F2D)
TB2D = torch.tensor(B2D)

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
    
    @abstractmethod
    def transition(self,x,u):
        pass
    
    @abstractmethod
    def predictC(self,C,Q):
        pass

class Linear3DFullActor(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,F3D,B3D_Full)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,F3D,B3D_Full)
    
class Linear3D2DActor(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,F3D,B3D_2D)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,F3D,B3D_2D)
    
class Linear3DFullActorT(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,TF3D,TB3D_Full)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,TF3D,TB3D_Full)
    
class Linear3D2DActorT(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def transition(self,x,u):
        return self._transition(x,u,TF3D,TB3D_2D)

    def predictC(self,C,Q):
        return self._predictC(C,Q,TF3D,TB3D_2D)
    
class Linear2DActor(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,F2D,B2D)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,F2D,B2D)
    
class Linear2DActorT(Actor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition(self,x,u):
        return self._transition(x,u,TF2D,TB2D)
    
    def predictC(self,C,Q):
        return self._predictC(C,Q,TF2D,TB2D)
    