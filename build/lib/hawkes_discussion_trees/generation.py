import numpy as np
import networkx as nx
from scipy import optimize
from copy import deepcopy

from .utils import * 


def weibull_poisson_times(a, b, alpha, T=None, limit_size=None, start_time = 0.0):   
    '''Generate arrival times of Poisson process with Weibull intensity. 
    Thinning algorithm of Lewis and Shedler [1979] was implemented.

    Two stopping conditions are implemented:

    'T': fixed time interval for generation - when the next arrival time overpasses T, 
    the generation stops;
    'limit_size': max number of arrivals in the generation - when the number of arrivals hits limit_size, 
    the generation stops.

    Input: 
        a: float                -- parameter; it is equal to the average number of comments to the root
        b: float                -- parameter; it represents a time scale of the generation process
        alpha: float            -- parameter; it is responsible for the shape of the Weibull pdf 
        T: float                -- stop condition; max duration of comment generation 
        limit_size: float       -- stop condition; max number of the generated comment arrival times

    Output: list of arrival times

    '''
    if T is None:
        T = 1000*b
    if limit_size is None:
        limit_size=1000*a

    def weibull_bound(t, a, b, alpha):
        '''Internal funtion for thinning procedure. Provides an upper bound on the intensity function'''
        lbd = 0
        if alpha>1:
            if t<b*((alpha-1)/alpha)**(1/alpha):
                lbd = (a*alpha/b)*((alpha-1)/alpha)**(1-1/alpha)*np.exp(-((alpha-1)/alpha))
            else:
                lbd = weibull_func(t, a, b, alpha)
        else:
            lbd = weibull_func(t, a, b, alpha)
        return lbd

    weibull_times_list = []
    t = start_time
    lbd = weibull_bound(t,a,b,alpha)
    while True:
        e = np.random.uniform(low=0, high=1)
        t += -np.log(e)/lbd
        U = np.random.uniform(low=0, high=1)
        if U < weibull_func(t,a,b, alpha)/lbd:
            weibull_times_list.append(t)
            lbd = weibull_bound(t, a, b, alpha)
        if len(weibull_times_list)>0:
            if t-weibull_times_list[-1]>T:
                break
        else:
            if t>T:
                break
        if len(weibull_times_list)>limit_size:
            return weibull_times_list
    return weibull_times_list


def lognormal_poisson_times(mu, sigma, avg_brnch, T=None, limit_size=None, start_time = 0.0): 
    '''Generate arrival times of Poisson process with LogNormal intensity. 
    Thinning algorithm of Lewis and Shedler [1979] was implemented.

    Two stopping conditions are implemented:

    'T': fixed time interval for generation - when the next arrival time overpasses T, 
    the generation stops;
    'limit_size': max number of arrivals in the generation - when the number of arrivals hits limit_size, 
    the generation stops.

    Input: 
        mu, sigma: float            -- parameters of LogNormal pdf
        avg_brnch: float            -- average branching number - constant multiplier of the intensity, 
        T: float                    -- time interval for generation, 
        limit_size: int             -- max size of the tree

    Output: 
        list of arrival times
    
    '''
    if T is None:
        T = 100*np.exp(mu-(sigma**2))
    if limit_size is None:
        limit_size=100*avg_brnch
        
    def lognorm_bound(t, mu, sigma, avg_brnch):
        '''Internal funtion for thinning procedure. Provides an upper bound on the intensity function'''
        if t<np.exp(mu-(sigma**2)):
            lbd = avg_brnch*(np.exp(sigma**2-mu)/(sigma*np.sqrt(2*np.pi)))*np.exp(-(sigma**2)/2)
        else:
            lbd = lognorm_func(t, mu, sigma, avg_brnch)
        return lbd
    
    lognorm_times_list = []
    t = start_time
    lbd = lognorm_bound(t, mu, sigma, avg_brnch)
    while True:
        e = np.random.uniform(low=0, high=1)
        t += -np.log(e)/lbd
        U = np.random.uniform(low=0, high=1)
        if U < lognorm_func(t, mu, sigma, avg_brnch)/lbd:
            lognorm_times_list.append(t)
            lbd = lognorm_bound(t, mu, sigma, avg_brnch)
        if len(lognorm_times_list)>0:
            if t-lognorm_times_list[-1]>T:
                break
        else:
            if t>T:
                break
        if len(lognorm_times_list) > limit_size:
            return lognorm_times_list
    return lognorm_times_list

def hawkes_comment_tree(root_params, offspring_params, T_root, T_comments, limit_tree_size=None, 
                          limit_root_size=None, limit_comments_size=None):
    '''
    Generate the Hawkes branching comment tree. By now only with Weibull root intensity and LogNormal 
    intensity for offsprings. Four stopping conditions are implemented:

        'T_root': fixed time interval for generation of comments to the root - when the next arrival time overpasses 
                T_root, the generation stops;
        'T_comments': fixed time interval for generation of comments to other comments - when the next arrival 
                time overpasses T_comments, the generation stops;
        'limit_tree_size': max number of arrivals of comments - when the number of comemnts hits limit_tree_size, 
                the generation stops.
        'limit_root_size': max number of comments to the root;
        'limit_comments_size': max number of comments to any another comment.
    
    Input: 
        root_params: list([a, b, alpha])     -- parameters for the root commenting process
        offspring_params: list([mu, sigma])  -- parameters for the commenting process of other comments
        T_root: float                        -- stop condition; max duration of comment generation to the root
        T_comments: float                    -- stop condition; max duration of comment generation to other comments
        limit_tree_size: int                 -- stop condition; max total number of comments in the tree
        limit_root_size: int                 -- stop condition; max number of extra comments to the root
        limit_comments_size: int             -- stop condition; max number of comments to any another comment
    
    Output: 
        g: nx.Graph()                        -- generated tree
        _: bool                              -- whether the generated tree did not exceed the limit_tree_size
    
    '''
    assert len(root_params) == 3 and len(offspring_params) == 3
    
    a, b, alpha = root_params
    mu, sigma, avg_brnch = offspring_params
    
    if limit_tree_size is None:  # arbitrary large number
        limit_tree_size = 100*a + 1000*avg_brnch
    
    if limit_root_size is None:
        limit_root_size=100*a
        
    if limit_comments_size is None:
        limit_comments_size=10*avg_brnch
        
    g = nx.Graph()
    node_index = 0
    offspring_node_index = 1
    g.add_node(node_index, created = 0, root=True)
    immigrant_times = weibull_poisson_times(a,b, alpha, T_root, limit_root_size)
    outcast = False
    for t in immigrant_times:
        node_index+=-1
        g.add_node(node_index, created = t, root = False)
        g.add_edge(0, node_index)
        tree_node_list = []
        offspring_times = lognormal_poisson_times(mu, sigma, avg_brnch, T_comments, limit_comments_size)
        offspring_times = [i+t for i in offspring_times]
        if len(offspring_times)>0:
            for t2 in offspring_times:
                g.add_node(offspring_node_index, created = t2, root = False)
                g.add_edge(node_index, offspring_node_index)
                tree_node_list.append(offspring_node_index)
                offspring_node_index+=1
        while len(tree_node_list)!=0:
            current_node = tree_node_list[0]
            del tree_node_list[0]
            t_offspring = g.node[current_node]['created']
            offspring_times = lognormal_poisson_times(mu, sigma, avg_brnch, T_comments, limit_comments_size)
            offspring_times = [i+t_offspring for i in offspring_times]
            if len(offspring_times)>0:
                for t2 in offspring_times:
                    g.add_node(offspring_node_index, created = t2, root = False)
                    g.add_edge(current_node, offspring_node_index)
                    tree_node_list.append(offspring_node_index)
                    offspring_node_index+=1
            if nx.number_of_nodes(g)>limit_tree_size:
                return g, False
    return g, True



def continue_hawkes_comment_tree(given_tree, start_time, root_params, offspring_params, T_root, 
                                 T_comments, limit_tree_size=None, limit_root_size=None, 
                                 limit_comments_size=None):
    '''Given an initial subtree of a discussion, generate the rest of tree using Hawkes branching process 
    with inferred parameters. Same stopping conditions as in *hawkes_comment_tree()*.

    Input:
        given_tree: nx.Graph()               -- initial subtree, from which to start generation
        start_time: float                    -- absolute time from which generation starts
        root_params: list([a, b, alpha])     -- parameters for the root commenting process
        offspring_params: list([mu, sigma])  -- parameters for the commenting process of other comments
        
        T_root: float                        -- stop condition; max duration of comment generation to the root
        T_comments: float                    -- stop condition; max duration of comment generation to other comments
        limit_tree_size: int                 -- stop condition; max total number of comments in the tree
        limit_root_size: int                 -- stop condition; max number of extra comments to the root
        limit_comments_size: int             -- stop condition; max number of comments to any another comment

    Output:
        g: nx.Graph()                        -- generated tree
        _: bool                              -- whether the generated tree did not exceed the limit_tree_size
    
    '''
    a,b, alpha = root_params
    mu, sigma, avg_brnch = offspring_params
    g = deepcopy(given_tree)
    root = get_root(g)
    root_comment_nodes = []
    node_index = max(g.nodes())
    existing_comment_nodes = []
    
    root_comment_nodes = [u for u in g.neighbors(root)]
    for v in g.nodes():
        if v not in root_comment_nodes and v != root:
            existing_comment_nodes.append(v)
        
    gen_comment_arrival_times = []
    while len(existing_comment_nodes)>0:
        comment_node = deepcopy(existing_comment_nodes[0])
        del existing_comment_nodes[0]
        comment_time = g.node[comment_node]['created']
        gen_comment_arrival_times.clear()
        gen_comment_arrival_times = lognormal_poisson_times(mu, sigma, avg_brnch, T = T_comments, 
                                                            limit_size = limit_comments_size,
                                                            start_time = start_time - comment_time)
        next_comment_arrival_times = [i+comment_time for i in gen_comment_arrival_times]
        generated_comment_nodes = []
        if len(next_comment_arrival_times)>0:
            for t in next_comment_arrival_times:
                node_index += 1
                g.add_node(node_index, created = t, root = False)
                g.add_edge(comment_node, node_index)
                generated_comment_nodes.append(node_index)
                
        while len(generated_comment_nodes)!=0:
            current_node = deepcopy(generated_comment_nodes[0])
            del generated_comment_nodes[0]
            comment_time = g.node[current_node]['created']
            gen_comment_arrival_times.clear()
            gen_comment_arrival_times = lognormal_poisson_times(mu, sigma, avg_brnch, T_comments, 
                                                 start_time = start_time - comment_time)
            next_comment_arrival_times = [i+comment_time for i in gen_comment_arrival_times]
            if len(next_comment_arrival_times)>0:
                for t2 in next_comment_arrival_times:
                    node_index += 1
                    g.add_node(node_index, created = t2, root = False)
                    g.add_edge(current_node, node_index)
                    generated_comment_nodes.append(node_index)
            if nx.number_of_nodes(g) > limit_tree_size:
                return g, False   
            
    next_root_comment_arrival_times = weibull_poisson_times(a,b, alpha, T = T_root, 
                                                            limit_size = limit_root_size,
                                                           start_time = start_time)
    for next_t in next_root_comment_arrival_times:
        node_index += 1
        g.add_node(node_index, created = next_t, root = False)
        g.add_edge(root, node_index)
        next_comment_nodes = []
        gen_comment_arrival_times = lognormal_poisson_times(mu, sigma, avg_brnch, T = T_comments, 
                                                           limit_size = limit_comments_size)
        gen_comment_arrival_times = [t + next_t for t in gen_comment_arrival_times]
        if len(gen_comment_arrival_times)>0:
            current_node = node_index
            for t2 in gen_comment_arrival_times:
                node_index += 1
                g.add_node(node_index, created = t2, root = False)
                g.add_edge(current_node, node_index)
                next_comment_nodes.append(node_index)
                
        while len(next_comment_nodes)!=0:
            current_node = next_comment_nodes[0]
            del next_comment_nodes[0]
            t_offspring = g.node[current_node]['created']
            gen_comment_arrival_times = lognormal_poisson_times(mu, sigma, avg_brnch, T = T_comments, 
                                                           limit_size = limit_comments_size)
            gen_comment_arrival_times = [t + next_t for t in gen_comment_arrival_times]
            if len(gen_comment_arrival_times)>0:
                for t2 in gen_comment_arrival_times:
                    node_index += 1
                    g.add_node(node_index, created = t2, root = False)
                    g.add_edge(current_node, node_index)
                    next_comment_nodes.append(node_index)
            if nx.number_of_nodes(g) > limit_tree_size:
                return g, False
    return g, True