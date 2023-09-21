# -*- coding: utf-8 -*-
from abc import ABC,abstractmethod

"""Facet class, encapsure Sensor"""
class Sensor(ABC):
        
    @abstractmethod
    def hx(self,x,xs):
        pass
    @abstractmethod
    def H(self,x,xs):
        pass
    @abstractmethod
    def sx(self,x,xs):
        pass
    @abstractmethod
    def S(self,x,xs):
        pass
        
    
__all__=['Sensor']