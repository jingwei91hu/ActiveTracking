# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, unicode_literals)

from scipy.interpolate import interp1d, splprep, splev
import numpy as np

def trajectory_generator_2D(points,T,per):
    N = points.shape[0]
   
    tck, u = splprep(points.T, u=None, s=1, k=2, per=per) 
 
    u = np.linspace(u.min(), u.max(), T)
    
    xs, ys = splev(u, tck, der=0)
    
    a = []
    vx = 0
    vy = 0
    for i in range(T-1):
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        ax = 2*(dx-vx)
        ay = 2*(dy-vy)
        vx += ax
        vy += ay
        a.append([ax,ay])
    a = np.array(a)
    return xs,ys,a

def trajectory_spiral_3D(p0,T,rotate):
    w = 0.15
    r = 5.
    xs = np.zeros(T)
    ys = np.zeros(T)
    zs = np.zeros(T)
    
    xs[0] = p0[0]
    ys[0] = p0[1]
    zs[0] = p0[2]
    
    t = np.arange(1,T)
    xs[1:] = r*np.sin(w*t) + xs[0]
    ys[1:] = ys[0] + rotate*0.005*(t**2)
    zs[1:] = r*np.sin(w*t) + zs[0]
    
    a = []
    vx = 0
    vy = 0
    vz = 0
    for i in range(T-1):
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        dz = zs[i+1] - zs[i]
        
        ax = 2*(dx-vx)
        ay = 2*(dy-vy)
        az = 2*(dz-vz)
        vx += ax
        vy += ay
        vz += az
        a.append([ax,ay,az])
    a = np.array(a)
    return xs,ys,zs,a


def trajectory_sin_2D(p0,T,rotate):
    w = 0.15
    r = 5.
    xs = np.zeros(T)
    ys = np.zeros(T)
    
    xs[0] = p0[0]
    ys[0] = p0[1]
    
    t = np.arange(1,T)
    xs[1:] = r*np.sin(w*t) + xs[0]
    ys[1:] = ys[0] + rotate*0.5*t
    
    a = []
    vx = 0
    vy = 0
    for i in range(T-1):
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        
        ax = 2*(dx-vx)
        ay = 2*(dy-vy)
        vx += ax
        vy += ay
        a.append([ax,ay])
    a = np.array(a)
    return xs,ys,a

def trajactory_Tangential_3D(p0,T,radius,angle):
    center = (p0[0]-radius,p0[1],p0[2])
    us = []
    vx = 0
    vy = 0
    px = p0[0]
    py = p0[1]
    xs = [px]
    ys = [py]
    for i in range(T):
        px1 = center[0]+radius*np.cos(i*angle/T)
        vx1 = px1-px
        ux = vx1-vx
        vx = vx1
        px = px1
        xs.append(px)
        
        py1 = center[1]+radius*np.sin(i*angle/T)
        vy1 = py1-py
        uy = vy1-vy
        vy = vy1
        us.append((ux,uy,0))
        py = py1
        ys.append(py)
    return np.array(xs),np.array(ys),np.zeros(len(xs)),np.array(us)