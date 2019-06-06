import numpy as np
import networkx as nx
from copy import deepcopy

def hawkes_loglikelihood(time_series, root_parameters, offspring_params):
    '''Return loglikelihood of the given timeseries to be from Hawkes model with given root commenting process 
    parameters and parameters of further commenting process.
    
    Input:
        time_series: list of floats             -- given time series to test
        root_parameters: list([a, b, alpha])    -- parameters for the root commenting process
        offspring_params: list([mu, sigma])     -- parameters for the further commenting process
    
    Output:
        logL: float -- loglikelihood value
        
    '''
    t_n = time_series[-1]
    [a, b, alpha] = root_parameters
    [mu, sigma, avg_brnch] = offspring_params
    logL = -a*(1-np.exp(-(t_n/b)**alpha))  # integral part of \mu(t)
    
    phi_sum = 0
    for t_i in time_series[:-1]:
        x1 = (np.log(t_n-t_i)-mu)/(np.sqrt(2)*sigma)
        logL += avg_brnch*(1/2 + (1/2)*erf(x1))  # integral part of each integral of \phi(t)
        logL += np.log((a*alpha/b)*(t_i/b)**(alpha-1)*np.exp(-(t_i/b)**alpha) + phi_sum)
        phi_sum += lognorm_func(t_n-t_i, mu, sigma, avg_brnch)  # sum of \phi(t) values under the log
    logL += np.log((a*alpha/b)*(t_n/b)**(alpha-1)*np.exp(-(t_n/b)**alpha) + phi_sum) # last value under log
    return logL

def extract_comment_arrival_times(tree, root=None):
    '''Given the discussion tree and its root, output the sorted list of comment arrival times to the root.
    
    Input: 
       tree: nx.Graph()                                -- discussion tree
    
    Output: 
        root_arrival_times: list of floats             -- comment arrival times to the root
        comment_arrival_times: list of floats          -- comment arrival times to other comments.
    '''
    if root is None:
        root = get_root(tree)
    
    # root comments
    root_arrival_times = []
    for u in tree[root]:
        root_arrival_times.append(tree.node[u]['created'])
    root_appearance_time = tree.node[root]['created']
    root_arrival_times = [(t-root_appearance_time) for t in root_arrival_times]
    root_arrival_times.sort()
    
    # other comment times
    comment_arrival_times = []
    sh_paths_dict = nx.shortest_path_length(tree, source=root)
    for u, d in sh_paths_dict.items():
        if d > 0:    # ensure that u is not the root
            if tree.degree(u)>1:     # check if there are other comments
                times_to_add = []
                for v in tree.neighbors(u):
                    if sh_paths_dict[v] > d:  # ensure to take only further comment arrival times
                        times_to_add.append(tree.node[v]['created'])
                times_to_add.sort()
                times_to_add = [(t-tree.node[u]['created']) for t in times_to_add]   # offset by the creation time of the parent comment
                comment_arrival_times += times_to_add
    comment_arrival_times.sort()
    return root_arrival_times, comment_arrival_times

def get_root(tree:'nx.Graph') -> 'node, else None':
    '''Find the root of the discussion tree.
    
    Input: 
        tree: nx.Graph()        -- discussion tree
    
    Output: 
        nx.node()               -- node
    '''
    for u in tree.nodes():
        if (tree.node[u]['root']):
            return u
    return None

def relabel_nodesto_int(tree):
    '''Return the tree with relabelled nodes as integers.

    Input:
        tree: nx.Graph()    -- discussion tree

    Output:
        nx.Graph()          -- return tree
    '''
    g = deepcopy(tree)
    g_out = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default')
    for u in g_out.nodes():
        if u > 0:
            g_out.node[u]['created'] = float(g_out.node[u]['created']-g_out.node[0]['created'])
        else:
            g_out.node[u]['created'] = 0.0
    return g_out

def truncate_tree_by_size(tree, truncate_size, root = None):
    '''Truncate the tree at the given size *truncate_size*.
    
    Input: 
        tree: nx.Graph()        -- discussion tree
        truncate_size: int      -- truncation size
    
    Output: 
        nx.Graph()              -- return tree
    '''
    g = deepcopy(tree)
    if root is None:
        root = get_root(g)
    
    nodes_by_time = []
    for u in g.nodes():
        t = (g.node[u]['created']-g.node[root]['created'])
        nodes_by_time.append((u,t))

    nodes_by_time.sort(key = lambda x: x[1])
    nodes_to_delete = [v for i, (v,t) in enumerate(nodes_by_time) if i > truncate_size]

    for u in nodes_to_delete:
        g.remove_node(u)
    g_out = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default')
    for u in g_out.nodes():
        if u > 0:
            g_out.node[u]['created'] = float(g_out.node[u]['created']-g_out.node[0]['created'])
        else:
            g_out.node[u]['created'] = 0.0
    return g_out

def truncate_tree_by_time(tree, truncate_time, root = None):
    '''Truncate the tree at the given time *truncate_time*.
    
    Input: 
        tree: nx.Graph()        -- discussion tree
        truncate_time: int      -- truncation time
    
    Output: 
        nx.Graph()              -- return tree
    '''
    g = deepcopy(tree)
    if root is None:
        root = get_root(g)
    
    nodes_to_delete = []
    for u in g.nodes():
        t = (g.node[u]['created']-g.node[root]['created'])
        if t > truncate_time:
            nodes_to_delete.append(u)
    for u in nodes_to_delete:
        g.remove_node(u)
    g_out = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default')
    for u in g_out.nodes():
        if u > 0:
            g_out.node[u]['created'] = float(g_out.node[u]['created']-g_out.node[0]['created'])
        else:
            g_out.node[u]['created'] = 0.0
    return g_out


def get_timeseries_from_tree(tree):
    '''Get time series of comments arrival from the discussion tree.
    
    Input: 
        tree: nx.Graph()                -- discussion tree
    
    Output: 
        timeline: list of floats        -- time series from tree
    '''
    timeline = []
    for u in tree.nodes():
        if (tree.node[u]['root']):
            root = u
        timeline.append(tree.node[u]['created'])
    min_t = tree.node[root]['created']
    timeline = [(i-min_t) for i in timeline if i-min_t>0]
    timeline.sort()
    return timeline

def weibull_func(t:'argument', a:'a', b:'b', alpha:'alpha') -> 'value: float': 
    '''Return Weibull pdf function $W(t,a,b,\alpha)$ evaluated at t.

    Parameters: 
        $a, b, alpha$

    Input: 
        t: float                            -- value at which function is evaluated
        [a, b, alpha]: list of float        -- parameters

    Returns: 
        f: float                            -- return value

    '''
    # a>0, b>0, alpha>0  --- Weibull
    f = (a*alpha/b)*(t/b)**(alpha-1)*np.exp(-(t/b)**alpha)
    return f

def lognorm_func(t:'argument', mu, sigma, avg_brnch): 
    '''Return value of LogNormal pdf function scaled by avg_brunch $avg_brnch*LN(t,\mu,\sigma)$ evaluated at t.
    

    Input: 
        t: float                                -- value at which function is evaluated
        [mu, sigma, avg_brnch]: list of float   -- parameters

    Output: 
        f: float                                -- return value
    
    '''
    if t>0:   #  if t == 0, then the function is formally undefined, we define it as a right limit value
        f = avg_brnch*(1/(t*sigma*np.sqrt(2*np.pi)))*np.exp(-(np.log(t)-mu)**2/(2*sigma**2))
    else:
        f = 0
    return f