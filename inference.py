import pickle
import networkx as nx
import numpy as np
from scipy import optimize
from scipy.special import erf, gamma
import warnings as _wn


'''Fit the Hawkes process given the comment arrival times to the root and to other comments. 
Please use *get_comment_arrival_times()* first to obtain the corresponding times.
'''
def mu(root_comment_arrivals, initial_guess = [50, 500, 1.8]):
    '''Fit $\mu(t)$ function given the comment arrival times to the root. 
    
    !!  For now $\mu(t)$ is hard-coded to be scaled Weibull pdf  !!
    
    Input: 
    	root_comment_arrivals: list of floats
    	initial_guess: list of floats [a,b,alpha]
    
    Output: 
    	parameters: [a,b,alpha]
    '''
    if len(root_comment_arrivals) < 20:
    	_wn.warn("The number of points is small. Inference results may be unstable.")
    def weibull_loglikelihood(var):  # var = (a,b,alpha)
        '''Weibull pdf loglikelihood function. '''
        t_n = root_comment_arrivals[-1]
        f = (-var[0]*(1-np.exp(-(t_n/var[1])**(var[2]))) + 
             len(root_comment_arrivals)*(np.log(var[0])+np.log(var[2])-(var[2])*np.log(var[1])))
        for t in root_comment_arrivals:
            f+= (var[2]-1)*np.log(t)-(t/var[1])**(var[2])
        return (-1)*f

    result = optimize.minimize(weibull_loglikelihood, initial_guess, method = 'L-BFGS-B',
                               bounds = ((0.0001, None), (0.0001, None), (0.0001, None)))
    fit_params = list(result.get('x'))
    return fit_params     # [a,b,alpha]

def phi(other_comment_arrivals, initial_guess = [4,2]):
    '''Fit $\phi(t)$ function given the comment arrival times to other comments. 
    
    !!  For now $\phi(t)$ is hard-coded to be scaled LogNormal pdf  !!
    
    Input: 
    	other_comment_arrivals: list of floats
   		initial_guess: list of floats [mu, sigma]
    
    Output: 
    	parameters: [mu, sigma]
    '''
    if len(other_comment_arrivals) < 20:
    	_wn.warn("The number of points is small. Inference results may be unstable.")
    def lognorm_loglikelihood(var): # var = [mu,sigma]
        '''LogNormal pdf loglikelihood function. '''
        t_n = other_comment_arrivals[-1]
        f = ((-1/2-(1/2)*erf((np.log(t_n)-var[0])/(np.sqrt(2)*var[1]))) + 
             len(other_comment_arrivals)*np.log(1/(var[1]*np.sqrt(2*np.pi))))
        for t in other_comment_arrivals:
            f += -(np.log(t)-var[0])**2/(2*var[1]**2)-np.log(t)
        return (-1)*f

    result = optimize.minimize(lognorm_loglikelihood, initial_guess, method = 'L-BFGS-B', 
                               bounds = ((0.0001, None), (0.0001,None)))
    fit_params = list(result.get('x'))
    return fit_params  # [mu, sigma]

def avg_branching(root_comment_arrivals, other_comment_arrivals):
    '''Average branching number evaluation from data. See Medvedev, Delvenne, Lambiotte [2018] for 
    further explanation.
    
    Input: 
    	root_comment_arrivals: list of floats
    	other_comment_arrivals: list of floats
    
    Output: 
    	avg_brnch: float
    '''
    avg_brnch = 1- len(root_comment_arrivals)/(len(root_comment_arrivals)+len(other_comment_arrivals))
    return avg_brnch