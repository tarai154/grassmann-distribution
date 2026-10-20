






import numpy as np


def generate_master_dummy(level_list, drop_first=True):
    """
    Generate all possible dummy pattern matrix specified in level_list.
    
    Parameters
    ----------
    level_list : one-dimensional numpy.ndarray or list, dtype=int
        Level of measurement of each column in the data matrix. Each integer is encoded as follows:
        0 for a continuous variable, 1 for a binary variable.
    drop_first : bool, optional
        Indicates whether to drop the column corresponding to the base category in dummy encoding. The default is True.

    Returns
    -------
    master_dummy : numpy.ndarray of shape (2**n_dummy, n_dummy)
        Generated dummy matrix containing 0 or 1.

    """
    
    
    level_list = np.asarray(level_list, dtype=int) 
    

    if level_list.size == 0 or np.sum(level_list != 0) == 0:
        return np.zeros((0, 0))


    if drop_first:
        level_base = 0
    else:
        level_base = 1


    master_dummy = np.zeros((1, np.abs(level_list[0]) + level_base))
    
    if drop_first == False:
        master_dummy[0, 0] = 1
    
    
    for i_level in range(level_base, np.abs(level_list[0]) + level_base):

        temp_matrix = np.zeros((1, np.abs(level_list[0]) + level_base))
        
        if level_list[0] > 0:
            temp_matrix[:, i_level] = 1
        else:
            temp_matrix[:, 0:(i_level + 1)] = 1
            
            
        master_dummy = np.vstack((master_dummy,
                                  temp_matrix))
        
        
            
    
    for i_feature in range(1, level_list.size):
        
        
        temp_matrix = np.zeros((master_dummy.shape[0], np.abs(level_list[i_feature]) + level_base))
        if drop_first == False:
            temp_matrix[:, 0] = 1
        
        
        stacked_array = np.hstack((master_dummy, temp_matrix))
        
        for i_level in range(level_base, np.abs(level_list[i_feature]) + level_base):
                
            
            temp_matrix = np.zeros((master_dummy.shape[0], np.abs(level_list[i_feature]) + level_base))
            
                    
            if level_list[i_feature] > 0:
                temp_matrix[:, i_level] = 1
            else:
                temp_matrix[:, 0:(i_level + 1)] = 1
                
                
            temp_stacked_array = np.hstack((master_dummy, temp_matrix))
            
            stacked_array = np.vstack((stacked_array,
                                       temp_stacked_array))
                            
            
            
        master_dummy = stacked_array.copy()
                        
    

    master_dummy = master_dummy.astype(np.bool_)
    

    
    return master_dummy






    
    





def simulate_dichotomized_gaussian(level_list, n_samples=1, random_state=None):
    """
    Generate a continuous and categorical and ordinal random dataset
    from correlated Gaussian distribution by dichotomization.

    Parameters
    ----------
    level_list : one-dimensional numpy.ndarray or list, dtype=int
        Level of measurement of each column. Each integer is encoded as follows:
        0 for a continuous variable, a positive integer for a categorical (nominal) variable,
        and a negative integer for an ordinal variable.
        For categorical and ordinal variables, the absolute value gives the number of categories.
    n_samples : int, optional
        Number of samples to simulate. The default is 1.
    random_state : int or None, optional
        Random seed for an initialization of model parameters. The default is None.

    Returns
    -------
    simulated_data : numpy.ndarray of shape (n_samples, n_mixture)
        Simulated data.

    """
    


    
    from scipy.stats import norm

    level_list = np.asarray(level_list)
    
    rng = np.random.default_rng(random_state)
    
    
    
    
    continuous_bool = np.array([], dtype=np.bool_)
    for level in level_list:
        if level == 0:
            continuous_bool = np.append(continuous_bool, True)
        else:
            continuous_bool = np.append(continuous_bool, np.full(np.abs(level), False))
    
    dummy_bool = np.logical_not(continuous_bool)

    n_dummy = np.sum(dummy_bool)
    n_continuous = np.sum(continuous_bool)
    n_mixture = n_continuous + n_dummy
    
    
    n_gauss = n_mixture - np.sum(np.abs(level_list[level_list < 0])) + np.sum(level_list < 0)
    
    
    U = rng.normal(loc=0, scale=1, size=n_gauss**2).reshape((n_gauss, n_gauss))
    
    U_tilde = U
    for j in range(1, n_gauss):
    
      for k in range(j):
    
        denominator = np.sum(U_tilde[:, k]**2)
        numerator = np.sum(U_tilde[:, k] * U[:, j])
        U_tilde[:, j] = U_tilde[:, j] - numerator / denominator * U_tilde[:, k]
    
    U_tilde = U_tilde @ np.diag(1/np.sqrt(np.diag(U_tilde.T @ U_tilde)))
    
    
    
    eigenvalues = rng.gamma(1, 1, size=n_gauss)
    
    
    Sigma = U_tilde @ np.diag(eigenvalues) @ U_tilde.T
    
    Sigma = np.sqrt(1/np.diag(Sigma)).reshape((-1,1)) * Sigma * np.sqrt(1/np.diag(Sigma)).reshape((1,-1))
    
    
    
    
    
    
    gaussian_data = rng.multivariate_normal(np.zeros(n_gauss), cov=Sigma, size=n_samples)
    simulated_data = np.zeros((n_samples, n_mixture))
    
    i_gauss = 0
    i_data = 0
    for i_feature in range(len(level_list)):
        
        
        temp_level = np.abs(level_list[i_feature])
        
        if temp_level == 0:
            simulated_data[:, i_data] = gaussian_data[:, i_gauss]
            i_gauss = i_gauss + 1
            i_data = i_data + 1
            
        
        
        elif level_list[i_feature] > 0:
            
            
            threshold_prob = rng.dirichlet(alpha=1*np.ones(1 + 1), size=temp_level)
            threshold_prob = threshold_prob[:, 0]
            
            temp_underlying_mu = norm.ppf(threshold_prob, loc=0, scale=1)            
            
            
            
            temp_gaussian_data = np.column_stack([gaussian_data[:, i_gauss:(i_gauss + temp_level)] + temp_underlying_mu.reshape((1,-1)),
                                                    np.zeros(n_samples)])
            
            argmax_list = np.argmax(temp_gaussian_data, axis=1)
            
            for i_level in range(temp_level):
                
                temp_bool = argmax_list == i_level
                simulated_data[temp_bool, i_data + i_level] = 1
            
            
            i_gauss = i_gauss + temp_level
            i_data = i_data + temp_level
            
            

            
            
        else:
        
            threshold_prob = rng.dirichlet(alpha=1*np.ones(temp_level + 1), size=1)
            threshold_prob = np.cumsum(threshold_prob.ravel())[:-1]
            
            
            temp_threshold = norm.ppf(threshold_prob, loc=0, scale=1)
            
                        
            temp_threshold = np.append(-np.inf, temp_threshold)
            temp_threshold = np.append(temp_threshold, np.inf)
            
            temp_dummy_data = np.zeros((n_samples, temp_level))
            for i_level in range(1, temp_level + 1):
                
                temp_dummy = np.zeros(temp_level)
                temp_dummy[0:i_level] = 1
                
                temp_bool = (temp_threshold[i_level] <= gaussian_data[:, i_gauss]) * (gaussian_data[:, i_gauss]  < temp_threshold[i_level + 1])
                
                temp_dummy_data[temp_bool, :] = temp_dummy.reshape((1, temp_level))
                
            

            
            simulated_data[:, i_data:(i_data + temp_level)] = temp_dummy_data
            
            i_gauss = i_gauss + 1
            i_data = i_data + temp_level
    
        
    
    return simulated_data


