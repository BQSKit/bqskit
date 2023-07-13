"""This package defines constant gates."""
from __future__ import annotations

from bqskit.utils.math import kron
import numpy as np
import itertools 
import jax 
jax.config.update("jax_enable_x64", True) # update to use with numpy only

p0=np.array([[1,0,0],
    [0,0,0],
    [0,0,0]])
p1=np.array([[0,0,0],
    [0,1,0],
    [0,0,0]])
p2=np.array([[0,0,0],
    [0,0,0],
    [0,0,1]])

pdict={0: p0, 1: p1, 2: p2}

def cgate(i,op):
    total_pairs=set((0,1,2))
    total_pairs.remove((i))
    result = np.zeros((9,9))
    for pair in total_pairs:
        result+=kron([pdict[pair],np.eye(3)])
    result+=kron([pdict[i],op])
    return result

def ccgate(vec,op):
    total_pairs=set(itertools.product((0,1,2),(0,1,2)))
    total_pairs.remove((vec[0],vec[1]))
    result = np.zeros((27,27))
    for pair in total_pairs:
        result+=kron([pdict[pair[0]],pdict[pair[1]],np.eye(3)])
    result+=kron([pdict[vec[0]],pdict[vec[1]],op])
    return result


def jaxkron(mlist):
    m=mlist[0]
    for i in range(1,len(mlist)):
        m=jax.numpy.kron(m,mlist[i])
    return m

def jaxccgate(i,j,op):
    total_pairs=set(itertools.product((0,1,2),(0,1,2)))
    total_pairs.remove((i,j))
    result = jax.numpy.zeros((27,27))
    for pair in total_pairs:
        result+=jaxkron([pdict[pair[0]],pdict[pair[1]],jax.numpy.eye(3)])
    result+=jaxkron([pdict[i],pdict[j],op])
    return result

def jaxcgate(i,op):
    total_pairs=set((0,1,2))
    total_pairs.remove((i))
    result = jax.numpy.zeros((9,9))
    for pair in total_pairs:
        result+=jaxkron([pdict[pair],jax.numpy.eye(3)])
    result+=jaxkron([pdict[i],op])
    return result

from bqskit.ir.gates.qutrit.constant import __all__ as __qutrit_all_constant__
from bqskit.ir.gates.qutrit.parameterized import __all__ as __qutrit_all_parameterized__

