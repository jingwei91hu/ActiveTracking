# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, unicode_literals)
from .actors import *

"""Facet class, encapsure Target vs Sensor"""



class Target3D(Linear3DFullActor):
    def __init__(self):
        super().__init__(n_static=0)

class Target3DT(Linear3DFullActorT):
    def __init__(self):
        super().__init__(n_static=0)

class Target2D(Linear2DActor):
    def __init__(self):
        super().__init__(n_static=0)

class Target2DT(Linear2DActorT):
    def __init__(self):
        super().__init__(n_static=0)
    