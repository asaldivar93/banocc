import pystan
import pickle
from stan_utility import utils as su
import arviz as az
import numpy as np
import pandas as pd

def run_banocc(C, file = 'BAnOCC.pkl', a = 0.5, b = 0.01, chains = 4, 
               iters = 50, thin = 1, init = [], control = []):
    
    warmup = int(np.floor(iters/2))
    
    if not isinstance(C, np.ndarray):
        print('Converted C to numpy')
        C = C.to_numpy()
    
    print('Checking C')
    check_c(C)
    banocc = pickle.load(open(file, 'rb'))
    
    n = 0*np.ones(C.shape[1])
    L = np.diag(10*np.ones(C.shape[1]))
    data = {'C':C, 'n':n, 'L':L, 'a':a, 'b':b, 'N':C.shape[0], 'P':C.shape[1]}
    if not init:
        init = [get_init(data) for b in range(chains)]
        print('Started model fitting')
        fit = banocc.sampling(data = data, chains = chains, iter = iters, init = init, warmup = warmup, refresh = np.floor(iters/20))
     
    su.check_treedepth(fit)
    su.check_energy(fit)
    su.check_div(fit)
    return fit

def get_init(data):
    init = {'m':np.random.multivariate_normal(data['n'], data['L']),
           'O':np.diag(1*np.ones(data['P'])),
           'lamb':np.random.gamma(data['a'], data['b'])}
    
    return init
    
    
class SampleError(Exception):
    def __init__(self, value):
         self.value = value
    def __str__(self):
         return repr(self.value)
        
class CompositionError(Exception):
    def __init__(self, value):
         self.value = value
    def __str__(self):
         return repr(self.value)
        
def check_c(C):
    try:
        if np.any((C.sum(axis = 1) - 1) > 1e-8):
            raise SampleError('Composition for some samples does not sum to 1')
        if np.any((1 - C.sum(axis = 1)) > 1e-8):
            raise CompositionError('Composition for some samples sums to less than 1')
    except SampleError:
        raise
    except CompositionError:
        raise
    