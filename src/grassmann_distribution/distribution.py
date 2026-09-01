

import time

import numpy as np
from scipy import optimize
import jax.numpy as jnp
from jax import value_and_grad, jit


from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data, check_is_fitted


        
        
        
class GrassmannDistribution(BaseEstimator):
    """
    Implementation of the Grassmann distribution.
    
    Parameters
    ----------
    n_trial : int, optional
        Number of random initializations for parameter estimation. The default is 1.
    tol : float, optional
        Convergence tolerance for parameter estimation.
        When tol is specified, the relevant solver-specific tolerance in ``scipy.optimize.minimize()`` is set to ``tol``.
        The default is 1e-12.
    maxiter : int, optional
        Maximum number of iterations for parameter estimation. The default is 10000.
    random_state : int or None, optional
        Random seed for an initialization of model parameters. The default is None.


    Attributes
    -------
    B_ : numpy.ndarray of shape (n_dummy, n_dummy)
        A row diagonally dominant matrix.
    C_ : numpy.ndarray of shape (n_dummy, n_dummy)
        A strictly row diagonally dominant matrix.
    Lambda_ : numpy.ndarray of shape (n_dummy, n_dummy)
        A Grassmann model parameter, Lambda = B C^{-1} + I, where I is an identity matrix.
    Sigma_ : numpy.ndarray of shape (n_dummy, n_dummy)
        A Grassmann model parameter, Sigma = Lambda^{-1}.    
    loglik_ : float
        Log-likelihood evaluated at the maximum likelihood estimate.
    n_features_in_ : int
        Number of features seen during ``fit()``.
        
    """    
    
    
    
    def __init__(self, n_trial=1, tol=1e-12, maxiter=10000, random_state=None):
        """
        Constructor of the Grassmann distribution.

        Parameters
        ----------
        n_trial : int, optional
            Number of random initializations for parameter estimation. The default is 1.
        tol : float, optional
            Convergence tolerance for parameter estimation.
            When tol is specified, the relevant solver-specific tolerance in ``scipy.optimize.minimize()`` is set to ``tol``.
            The default is 1e-12.
        maxiter : int, optional
            Maximum number of iterations for parameter estimation. The default is 10000.
        random_state : int or None, optional
            Random seed for an initialization of model parameters. The default is None.

        """
        
        self.n_trial = n_trial
        self.tol = tol
        self.maxiter = maxiter
        self.random_state = random_state

        
    
    

    def fit(self, X, y=None):
        """
        Fit the model to the data using maximum likelihood estimation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dummy)
            Binary training data containing 0 and 1. NaN values are allowed and treated as missing values.
            NumPy arrays and pandas DataFrames are supported.
        y : None, optional
            Ignored. The default is None.

        Raises
        ------
        ValueError
            If ``X`` contains values other than 0, 1, or NaN.
            If the number of the variables in ``X`` is less than 2.

        Returns
        -------
        self : GrassmannDistribution
            Fitted model.

        """
                
        
        X = validate_data(self, X, reset=True, ensure_all_finite="allow-nan")
        
        valid_bool = np.isnan(X) | (X == 0) | (X == 1)
        if not np.all(valid_bool):
            raise ValueError("X must contain only 0, 1, or Nan.")
        
        
        if X.shape[1] < 2:
            raise ValueError("At this point, the implementation assumes at least two variables.")
        
        
        

        n_trial = self.n_trial
        tol = self.tol
        maxiter = self.maxiter
        random_state = self.random_state


        self.n_samples_ = X.shape[0]
        self.n_dummy_ = X.shape[1]
        
        
        
                
        import warnings
        import jax

        if not jax.config.read("jax_enable_x64"):
            
            warnings.warn("JAX 64-bit precision is required for reliable parameter estimation.\n"
                          "Enable JAX X64 mode before fitting the model with \n"
                          '`jax.config.update("jax_enable_x64", True)`.', RuntimeWarning, stacklevel=2)



        
        
        
        
        p_list, message, status, fun, n_iter, elapsed_time = _trial_iter(X, n_trial=n_trial, tol=tol, maxiter=maxiter, random_state=random_state)
        
        
        self.p_list_ = p_list
        
        
        
        
        B, C, Lambda, Sigma = _reconstruct_parameters(self.p_list_, self.n_dummy_)
        
        self.B_ = np.asarray(B)
        self.C_ = np.asarray(C)
        self.Lambda_ = np.asarray(Lambda)
        self.Sigma_ = np.asarray(Sigma)
        
        
        self.message_ = message
        self.status_ = status
        self.fun_ = fun
        self.n_iter_ = n_iter
        self.elapsed_time_ = elapsed_time
        
        
        
        
        self.loglik_ = - self.fun_ * self.n_samples_
        
                
        
        return self    
    
    
    
    
    
    
    @classmethod
    def from_parameters(cls, Sigma):
        """
        Create a GrassmannDistribution instance from specified model parameters.

        Parameters
        ----------
        Sigma : numpy.ndarray of shape (n_dummy, n_dummy)
            Parameter matrix Sigma.

        Raises
        ------
        ValueError
            If ``Sigma`` is not a square matrix.

        Returns
        -------
        model : GrassmannDistribution
            Distribution with the specified parameters.

        """
        
        Sigma = np.asarray(Sigma, dtype=np.float64)
        
        if (Sigma.ndim != 2) or (Sigma.shape[0] != Sigma.shape[1]):
            raise ValueError("Sigma must be a square matrix.")
            
        model = cls()
        model.Sigma_ = Sigma.copy()
        model.n_features_in_ = Sigma.shape[0]
        model.n_dummy_ = Sigma.shape[0]
        
        
        return model
    
    
    
    
    
    
        
    def score_samples(self, X):
        """
        Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dummy)
            Binary data for which the log-likelihood is computed. NaN values are allowed and treated as missing values.
            NumPy arrays and pandas DataFrames are supported.

        Raises
        ------
        ValueError
            If ``X`` contains values other than 0, 1, or NaN.
            If the model produces a negative joint probability.

        Returns
        -------
        fval_samples : numpy.ndarray of shape (n_samples, )
            Log-likelihood of each sample under the current model.

        """
        
        check_is_fitted(self, 'Sigma_')
        
        
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
                
        valid_bool = np.isnan(X) | (X == 0) | (X == 1)
        if not np.all(valid_bool):
            raise ValueError("X must contain only 0, 1, or Nan.")
                
        
        
        Sigma = self.Sigma_
        n_dummy = self.n_dummy_
        
        
        
        iszero_dummy = (X == 0)
        isnan_dummy = np.isnan(X)
        
        
        Sigma_array = np.einsum('ij,ik,jk->ijk', iszero_dummy, iszero_dummy, np.identity(n_dummy)) \
                      + np.einsum('ij,jk->ijk', 1 - 2*iszero_dummy, Sigma)
        
        
            
        Sigma_array = np.einsum('ij,ik,jk->ijk', isnan_dummy, isnan_dummy, np.identity(n_dummy)) \
                      + Sigma_array * np.einsum('ij,ik->ijk', np.logical_not(isnan_dummy), np.logical_not(isnan_dummy))
            
            
        
        
        sign_list, log_joint_list = np.linalg.slogdet(Sigma_array)
            
        
        
        
        
        if not np.all(sign_list >= 0):
            raise ValueError("The model produces a negative joint probability.")
        
        
        
        fval_samples = log_joint_list
        
    
        
        return fval_samples
        
        
        
        
        
        


    def score(self, X, y=None):
        """
        Compute the average log-likelihood over the samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dummy)
            Binary data for which the log-likelihood is computed. NaN values are allowed and treated as missing values.
            NumPy arrays and pandas DataFrames are supported.
        y : None, optional
            Ignored. The default is None.

        Returns
        -------
        float
            Average log-likelihood over the samples under the current model.

        """
        
        return float(np.mean(self.score_samples(X)))
    
    
    

    def logpmf(self, X):
        """
        Compute the logarithm of the probability mass function of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dummy)
            Binary data for which the logarithm of the probability mass function is computed. NaN values are allowed and treated as missing values.
            NumPy arrays and pandas DataFrames are supported.

        Returns
        -------
        numpy.ndarray of shape (n_samples, )
            Logarithm of the probability mass function of each sample under the current model.

        """
            
        return self.score_samples(X)
        
    
    
    
    def pmf(self, X):
        """
        Compute the probability mass function of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dummy)
            Binary data for which the probability mass function is computed. NaN values are allowed and treated as missing values.
            NumPy arrays and pandas DataFrames are supported.

        Returns
        -------
        numpy.ndarray of shape (n_samples, )
            Probability mass function of each sample under the current model.

        """
        
        return np.exp(self.logpmf(X))
        
    
    
    


    def sample(self, n_samples=1, random_state=None):
        """
        Generate random binary samples from the Grassmann distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. The default is 1.
        random_state : int or None, optional
            Random seed for random sampling. The default is None.

        Raises
        ------
        ValueError
            If the model produces a conditional probability outside the range [0, 1].

        Returns
        -------
        generated_data : numpy.ndarray of shape (n_samples, n_dummy)
            Generated binary data with elements equal to 0 or 1.

        """
        
        
        
        check_is_fitted(self, 'Sigma_')
        
        
        Sigma = self.Sigma_
        n_dummy = self.n_dummy_


        
        rng = np.random.default_rng(random_state)
        eps = 1e-15
            
        generated_data = np.zeros((n_samples, n_dummy), dtype=int)
        for i in range(n_samples):
    
            temp_Sigma = Sigma.copy()
            for j in range(n_dummy):
                
                temp_probability = temp_Sigma[0, 0]
                if not (0 - eps <= temp_probability <= 1 + eps):
                    raise ValueError("The model produces an invalid conditional probability (prob < 0 or prob > 1).")
                
                if not (0 <= temp_probability <= 1):
                    import warnings
                    warnings.warn("The conditional probability is slightly outside the range [0, 1], possibly due to numerical error.", RuntimeWarning)
                    
                    
                    
                temp_binary = rng.binomial(n=1, p=np.clip(temp_probability, 0, 1))
                generated_data[i, j] = temp_binary
    
                temp_Sigma = temp_Sigma[1:, 1:] - temp_Sigma[1:, 0:1] @ np.linalg.inv(temp_Sigma[0:1, 0:1] - (1 - temp_binary)) @ temp_Sigma[0:1, 1:]
    




        return generated_data
        
        



            

    def get_mean(self):
        """
        Compute the mean with the current model.

        Returns
        -------
        numpy.ndarray of shape (n_dummy, )
            The mean of each feature under the current model.

        """
        
        check_is_fitted(self, 'Sigma_')

        return np.diag(self.Sigma_)
    
    
    
    
    
    def get_corr(self):
        """
        Compute the correlation with the current model.

        Raises
        ------
        ValueError
            If at least one variable has zero variance.

        Returns
        -------
        rho_model : numpy.ndarray of shape (n_dummy, n_dummy)
            The correlation among features under the current model.

        """
        
        check_is_fitted(self, 'Sigma_')
        
        Sigma = self.Sigma_
        n_dummy = self.n_dummy_
        
        
        temp_mean = self.get_mean()
        
        if not np.all((temp_mean > 0) & (temp_mean < 1)):
            raise ValueError("Correlation is undefined because at least one variable has zero variance.")
        
        
        temp_var = temp_mean * (1 - temp_mean)
        rho_model = - Sigma * Sigma.T / np.sqrt(temp_var.reshape((-1,1)) @ temp_var.reshape((1,-1)))
        
        rho_model = rho_model - np.diag(np.diag(rho_model)) + np.identity(n_dummy)
        

    
        
        return rho_model






    def plot_corr(self):
        """
        Plot the correlation heatmap with the current model.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib Axes containing the correlation heatmap.

        """
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
    
        rho_model = self.get_corr()
    
    
        fig = plt.figure(dpi=350)
        ax = fig.add_subplot(111)
        
        sns.heatmap(rho_model, annot=True, fmt='.2f', cmap='jet', vmin=-0.7, vmax=0.7, ax=ax)
        
        
        
        return ax



















def _par_initial(n_dummy, rng=None):
    """
    Generate initial model parameters randomly.

    Parameters
    ----------
    n_dummy : int
        Number of binary variables.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created using ``numpy.random.default_rng()``.
        The default is None.
        
    Returns
    -------
    p_list : one-dimensional numpy.ndarray, dtype=float
        Optimizer-level model parameters.

    """
    

    if rng == None:
        rng = np.random.default_rng()
    
        
    p_list = np.abs(rng.normal(loc=0, scale=1/2, size=2*2*(n_dummy**2 - n_dummy)))


    return p_list



    






def _reconstruct_parameters(p_list, n_dummy):
    """
    Construct the Grassmann model parameters from optimizer-level model parameters.

    Parameters
    ----------
    p_list : one-dimensional numpy.ndarray
        Optimizer-level model parameters.
    n_dummy : int
        Number of binary variables.

    Returns
    -------
    B : jax.Array of shape (n_dummy, n_dummy)
        A row diagonally dominant matrix.
    C : jax.Array of shape (n_dummy, n_dummy)
        A strictly row diagonally dominant matrix.
    Lambda : jax.Array of shape (n_dummy, n_dummy)
        A Grassmann model parameter, Lambda = B C^{-1} + I, where I is an identity matrix.
    Sigma : jax.Array of shape (n_dummy, n_dummy)
        A Grassmann model parameter, Sigma = Lambda^{-1}.
        
    """
    

    b_double_list = p_list[0:2*(n_dummy**2 -n_dummy)]
    c_double_list = p_list[2*(n_dummy**2 -n_dummy):]
    
    
    
    
    b_abs_list = b_double_list[0:(n_dummy**2 - n_dummy)] + b_double_list[(n_dummy**2 - n_dummy):]
    c_abs_list = c_double_list[0:(n_dummy**2 - n_dummy)] + c_double_list[(n_dummy**2 - n_dummy):]
    
    
    b_list = b_double_list[0:(n_dummy**2 - n_dummy)] - b_double_list[(n_dummy**2 - n_dummy):]
    c_list = c_double_list[0:(n_dummy**2 - n_dummy)] - c_double_list[(n_dummy**2 - n_dummy):]
    
    
    
    
    linear_index = jnp.arange(n_dummy**2)
    row_indices = linear_index // n_dummy
    column_indices = linear_index % n_dummy
    
    
    off_diag_indices = np.where((np.identity(n_dummy) == 0).ravel())[0]
    off_diag_row_indices = row_indices[off_diag_indices]
    off_diag_column_indices = column_indices[off_diag_indices]
    
    
    B = jnp.zeros((n_dummy, n_dummy))
    B = B.at[off_diag_row_indices, off_diag_column_indices].set(b_list)
    B = jnp.diag(jnp.sum(b_abs_list.reshape((n_dummy, n_dummy - 1)), axis=1)) + B
    

    C = jnp.zeros((n_dummy, n_dummy))
    C = C.at[off_diag_row_indices, off_diag_column_indices].set(c_list)
    C = jnp.diag(1e-8 + (1 + 1e-8)*jnp.sum(c_abs_list.reshape((n_dummy, n_dummy - 1)), axis=1)) + C
    
    
    

    Sigma = jnp.linalg.solve((B + C).T, C.T).T
    Lambda = jnp.linalg.solve(C.T, (B + C).T).T    
    
    
    
    return B, C, Lambda, Sigma








def _negative_log_likelihood(p_list, qqq, unique_dummy):
    """
    Compute the average negative log-likelihood over the samples for the given model parameters.

    Parameters
    ----------
    p_list : one-dimensional numpy.ndarray
        Optimizer-level model parameters.
    qqq : one-dimensional numpy.ndarray, dtype=float
        Observed relative frequencies corresponding to the unique dummy-variable patterns.
        The elements of ``qqq`` must sum to one.
    unique_dummy : numpy.ndarray of shape (qqq.size, n_dummy)
        Unique dummy-variable patterns in the binary data.

    Returns
    -------
    fval : float
        Average negative log-likelihood over the samples for the specified model parameters.

    """

    n_dummy = unique_dummy.shape[1]
        
    
    B, C, Lambda, Sigma = _reconstruct_parameters(p_list, n_dummy)
    
        
    
    
    iszero_dummy = unique_dummy == 0
    isnan_dummy = jnp.isnan(unique_dummy)
    
    
    Sigma_array = jnp.einsum('ij,ik,jk->ijk', iszero_dummy, iszero_dummy, jnp.identity(n_dummy)) \
                  + jnp.einsum('ij,jk->ijk', 1 - 2*iszero_dummy, Sigma)
    
    
    Sigma_array = jnp.einsum('ij,ik,jk->ijk', isnan_dummy, isnan_dummy, jnp.identity(n_dummy)) \
                  + Sigma_array * jnp.einsum('ij,ik->ijk', jnp.logical_not(isnan_dummy), jnp.logical_not(isnan_dummy))
        
    
    
    _, log_det_Sigma = jnp.linalg.slogdet(Sigma_array)
        
    fval = - jnp.sum(qqq * log_det_Sigma)

    
    
    return fval




_negative_log_likelihood_and_grad = jit(value_and_grad(_negative_log_likelihood))











def _trial_iter(X, n_trial=1, tol=1e-12, maxiter=10000, random_state=None):
    """
    Generate random initialization of model parameters and perform optimization for each random initialization.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_dummy)
        Binary training data containing 0 and 1. NaN values are allowed and treated as missing values.
        NumPy arrays and pandas DataFrames are supported.
    n_trial : int, optional
        Number of random initializations for parameter estimation. The default is 1.
    tol : float, optional
        Convergence tolerance for parameter estimation.
        When tol is specified, the relevant solver-specific tolerance in ``scipy.optimize.minimize()`` is set to tol.
        The default is 1e-12.
    maxiter : int, optional
        Maximum number of iterations for parameter estimation. The default is 10000.
    random_state : int or None, optional
        Random seed for an initialization of model parameters. The default is None.

    Raises
    ------
    RuntimeError
        If all optimization trials failed.

    Returns
    -------
    p_list : one-dimensional numpy.ndarray, dtype=float
        Best optimizer-level model parameters.
    message : str
        Description of the cause of the optimizer termination for the best trial.
    status : int
        Termination status code of the optimizer for the best trial.
    fun : float
        Best average negative log-likelihood over the samples.
    n_iter : int
        Number of iterations performed by the optimizer for the best trial.
    elapsed_time : float
        Elapsed optimization time in seconds for the best trial.

    """
    
    
    
    
    
    rng = np.random.default_rng(random_state)
    n_dummy = X.shape[1]
    
    
    X_encoded = X.copy()
    X_encoded[np.isnan(X_encoded)] = 9999
    
    unique_dummy, nnn = np.unique(X_encoded, return_index=False, return_inverse=False, return_counts=True, axis=0)
    
    qqq = nnn/np.sum(nnn)    
    
    
    
    unique_dummy = unique_dummy.astype(np.float64)
    unique_dummy[unique_dummy == 9999] = np.nan
    
    
    
    bounds = [(0, None) for iii in range(_par_initial(n_dummy).size)]





    
    
    message_list = np.array([], dtype=object)
    result_list = np.empty((0, 5))
    p_matrix = np.empty((0, _par_initial(n_dummy).size))
    error_list = []
    
    for i_trial in range(n_trial):
        
        p_list = _par_initial(n_dummy, rng)
        
        
        try:
            
            
            ti = time.perf_counter()
            nit = 0
            
                
            
                    
            result = optimize.minimize(_negative_log_likelihood_and_grad, x0=p_list, args=(qqq, unique_dummy),
                                        jac=True,
                                        bounds=bounds,
                                        method='SLSQP', options={'ftol': tol, 'maxiter': maxiter,})
    
            nit = nit + result.nit
            
            
            tf = time.perf_counter()
            
            
            
            
    
    
            message_list = np.append(message_list, result.message)
            
            result_list = np.vstack((result_list,
                                      np.array([result.success,
                                                result.status,
                                                result.fun,
                                                nit,
                                                tf - ti]),
                                      ))
            
            p_matrix = np.vstack((p_matrix, result.x.copy()))
            
        
        
        
        
        except (ValueError, np.linalg.LinAlgError) as error:
            error_list.append(error)

        
        
        
        
    
    else:
        if result_list.shape[0] == 0:
            raise RuntimeError("All optimization trials failed.")
    
    
    
    
    
    
        
    
    temp_bool = (result_list[:, 1] == 0) | (result_list[:, 1] == 9)
    message_list = message_list[temp_bool]
    result_list = result_list[temp_bool, :]
    p_matrix = p_matrix[temp_bool, :]
    
        
    if result_list.shape[0] == 0:
        raise RuntimeError("Optimization failed.")    
        
        
    temp_index = np.lexsort((result_list[:, 2], result_list[:, 1]))
    message_list = message_list[temp_index]
    result_list = result_list[temp_index, :]
    p_matrix = p_matrix[temp_index, :]
    
    
    
    
    p_list = p_matrix[0, :]
    message = message_list[0]
    success = result_list[0, 0]
    status = result_list[0, 1]
    fun = result_list[0, 2]
    n_iter = result_list[0, 3]
    elapsed_time = result_list[0, 4]
    
    
    if success == False:
        import warnings
        warnings.warn("Optimization failed to converge.", RuntimeWarning, stacklevel=2)
        
        
    
    return p_list, message, int(status), float(fun), int(n_iter), float(elapsed_time)







