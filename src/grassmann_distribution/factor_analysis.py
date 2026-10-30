



from .utils import generate_master_dummy



import time

import numpy as np
from scipy import optimize










from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data, check_is_fitted


class FactorAnalysis(TransformerMixin, BaseEstimator):
    """
    Implementation of Grassmann factor analysis.

    Parameters
    ----------
    level_list : one-dimensional numpy.ndarray or list, dtype=int
        Level of measurement of each column in the data matrix. Each integer is encoded as follows:
        0 for a continuous variable, 1 for a binary variable.
    norm_constraint : bool, optional
        Whether to impose the equal-norm constraint on the factor loading matrices.
        If True, the row norms of the dimensionless factor loading matrices are constrained to be equal across all observed variables.
        If False, parameter estimation may yield improper solutions. The default is True.
    n_components : int, optional
        Number of latent dimensions. The default is 2.
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
    continuous_intercept_ : numpy.ndarray of shape (n_continuous, )
        Mean parameters for continuous variables.
    noise_variance_ : numpy.ndarray of shape (n_continuous, )
        Unique variances for continuous variables.
    continuous_components_ : numpy.ndarray of shape (n_continuous, n_components)
        Factor loading matrix for continuous variables.
    dummy_intercept_ : numpy.ndarray of shape (n_dummy, )
        Bias parameters for dummy variables.
    dummy_components_ : numpy.ndarray of shape (n_dummy, n_components)
        Factor loading matrix for dummy variables.
    c_z_ : numpy.ndarray of shape (n_mixture, )
        Row norms of the dimensionless factor loading matrices.
    loglik_ : float
        Log-likelihood evaluated at the maximum likelihood estimate.
    n_features_in_ : int
        Number of features seen during ``fit()``.

    """







    def __init__(self, level_list=None, norm_constraint=True, n_components=2, n_trial=1, tol=1e-12, maxiter=10000, random_state=None):
        """
        Constructor of the Grassmann factor analysis.

        Parameters
        ----------
        level_list : one-dimensional numpy.ndarray or list, dtype=int
            Level of measurement of each column in the data matrix. Each integer is encoded as follows:
            0 for a continuous variable, 1 for a binary variable.
        norm_constraint : bool, optional
            Whether to impose the equal-norm constraint on the factor loading matrices.
            If True, the row norms of the dimensionless factor loading matrices are constrained to be equal across all observed variables.
            If False, parameter estimation may yield improper solutions. The default is True.
        n_components : int, optional
            Number of latent dimensions. The default is 2.
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

        self.level_list = level_list
        self.n_components = n_components
        self.norm_constraint = norm_constraint
        self.n_trial = n_trial
        self.tol = tol
        self.maxiter = maxiter
        self.random_state = random_state









    def fit(self, X, y=None):
        """
        Fit the FactorAnalysis model to the data using maximum likelihood estimation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_mixture)
            Training data with mixed continuous and binary variables, where binary variables are encoded as 0 or 1.
            NaN values are allowed and treated as missing values.
            NumPy arrays and pandas DataFrames are supported.
        y : None, optional
            Ignored. The default is None.

        Raises
        ------
        ValueError
            If dummy data in ``X`` contains values other than 0, 1, or NaN.
            If ``norm_constraint`` is not a bool.

        Returns
        -------
        self : FactorAnalysis
            Fitted model.

        """


        n_trial = self.n_trial
        tol = self.tol


        if np.sum(self.level_list) == None:
            self.level_list = np.zeros((X.shape[1]), dtype=np.int_)

        level_list = self.level_list
        n_components = self.n_components


        continuous_bool = np.array([], dtype=np.bool_)
        for level in level_list:
            if level == 0:
                continuous_bool = np.append(continuous_bool, True)
            else:
                continuous_bool = np.append(continuous_bool, np.full(np.abs(level), False))

        dummy_bool = np.logical_not(continuous_bool)
        n_continuous = np.sum(continuous_bool)
        n_dummy = np.sum(dummy_bool)




        X = validate_data(self, X, reset=True, ensure_all_finite="allow-nan")

        dummy_data = X[:, dummy_bool]

        valid_bool = np.isnan(dummy_data) | (dummy_data == 0) | (dummy_data == 1)
        if not np.all(valid_bool):
            raise ValueError("Dummy data must contain only 0, 1, or Nan.")





        self.continuous_bool = continuous_bool
        self.n_continuous = n_continuous
        self.n_dummy = n_dummy
        self.n_features_in_ = n_continuous + n_dummy

        self.n_samples_ = X.shape[0]

        norm_constraint = self.norm_constraint



        import warnings
        import jax

        if not jax.config.read("jax_enable_x64"):

            warnings.warn("JAX 64-bit precision is required for reliable parameter estimation.\n"
                          "Enable JAX X64 mode before fitting the model with \n"
                          '`jax.config.update("jax_enable_x64", True)`.', RuntimeWarning, stacklevel=2)



        if type(norm_constraint) != bool:
            raise TypeError("``norm_constraint`` must be a boolean value.")



        maxiter = self.maxiter
        random_state = self.random_state



        p_list, status, fun, n_iter, elapsed_time, max_abs_grad, n_hess_minus_eigvals = _trial_iter(level_list, n_components, X, n_trial=n_trial, tol=tol, maxiter=maxiter, random_state=random_state, norm_constraint=norm_constraint)


        self.p_list_ = p_list

        self.status_ = status
        self.fun_ = fun
        self.n_iter_ = n_iter
        self.elapsed_time_ = elapsed_time
        self.max_abs_grad_ = max_abs_grad





        if norm_constraint == True:
            mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_norm(p_list, n_continuous, n_dummy, n_components)
        else:
            mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_naive(p_list, n_continuous, n_dummy, n_components)


        mu_x = np.asarray(mu_x)
        Psi = np.asarray(Psi)
        W = np.asarray(W)
        b_list = np.asarray(b_list)
        G = np.asarray(G)
        c_z = np.asarray(c_z)


        W, G, eigen_values = _rotate_loadings(p_list, level_list, n_components, norm_constraint)



        self.continuous_intercept_ = mu_x.flatten()
        self.dummy_intercept_ = b_list

        self.continuous_components_ = W
        self.dummy_components_ = G
        self.c_z_ = c_z

        self.noise_variance_ = np.diag(Psi)


        self.contribution_ratio_ = eigen_values / np.sum(eigen_values)

        self.loglik_ = - self.fun_ * self.n_samples_


        return self










    def transform(self, X):
        """
        Apply dimensionality reduction to the data using the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_mixture)
            Data with mixed continuous and binary variables, where binary variables are encoded as 0 or 1.
            NaN values are allowed and treated as missing values.
            NumPy arrays and pandas DataFrames are supported.

        Returns
        -------
        factor_scores : numpy.ndarray of shape (n_samples, n_components)
            Transformed array of factor scores.

        """

        continuous_bool = self.continuous_bool
        dummy_bool = np.logical_not(continuous_bool)


        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")

        dummy_data = X[:, dummy_bool]

        valid_bool = np.isnan(dummy_data) | (dummy_data == 0) | (dummy_data == 1)
        if not np.all(valid_bool):
            raise ValueError("Dummy data must contain only 0, 1, or Nan.")




        level_list = self.level_list
        n_components = self.n_components
        norm_constraint = self.norm_constraint

        p_list = self.p_list_


        factor_scores = _transform_data(p_list, level_list, n_components, X, norm_constraint=norm_constraint)



        return factor_scores
















    def score_samples(self, X):
        """
        Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_mixture)
            Data with mixed continuous and binary variables for which the log-likelihood is computed.
            NaN values are allowed and treated as missing values.
            NumPy arrays and pandas DataFrames are supported.

        Raises
        ------
        ValueError
            If dummy data in ``X`` contains values other than 0, 1, or NaN.

        Returns
        -------
        log_joint_list : numpy.ndarray of shape (n_samples, )
            Log-likelihood of each sample under the current model.

        """


        continuous_bool = self.continuous_bool
        dummy_bool = np.logical_not(continuous_bool)
        n_continuous = self.n_continuous
        n_dummy = self.n_dummy
        n_components = self.n_components

        level_list = self.level_list
        norm_constraint = self.norm_constraint

        p_list = self.p_list_



        master_dummy = generate_master_dummy(level_list)



        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")

        dummy_data = X[:, np.logical_not(continuous_bool)]

        valid_bool = np.isnan(dummy_data) | (dummy_data == 0) | (dummy_data == 1)
        if not np.all(valid_bool):
            raise ValueError("Dummy data must contain only 0, 1, or Nan.")




        if norm_constraint == True:
            mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_norm(p_list, n_continuous, n_dummy, n_components)
        else:
            mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_naive(p_list, n_continuous, n_dummy, n_components)




        mu_x = np.asarray(mu_x)
        Psi = np.asarray(Psi)
        W = np.asarray(W)
        b_list = np.asarray(b_list)
        G = np.asarray(G)
        c_z = np.asarray(c_z)



        Sigma_x = Psi + W @ W.T








        completed_data, bincount_index_list = _complete_nan_data(X, level_list)

        completed_data_continuous = completed_data[:, continuous_bool]
        completed_data_dummy = completed_data[:, dummy_bool]




        if n_dummy != 0:

            from scipy.special import logsumexp

            Q_a = np.sum(master_dummy.T * ((np.diag(b_list) + 0.5 * G @ G.T) @ master_dummy.T), axis=0)

            exponent_dummy = np.sum(completed_data_dummy.T * ((np.diag(b_list) + 0.5 * G @ G.T) @ completed_data_dummy.T), axis=0)

            log_Z_dummy = logsumexp(Q_a)

        else:
            exponent_dummy = 0

            log_Z_dummy = 0





        nan_pattern_matrix_continuous, nan_pattern_index_per_continuous, nan_nnn_continuous = np.unique(np.isnan(completed_data_continuous), return_inverse=True, return_counts=True, axis=0)

        non_nan_pattern_matrix_continuous = np.logical_not(nan_pattern_matrix_continuous).astype(np.int_)

        n_continuous_list = np.sum(non_nan_pattern_matrix_continuous, axis=1)

        Sigma_OO_array = np.einsum('aj,ak,jk->ajk', non_nan_pattern_matrix_continuous, non_nan_pattern_matrix_continuous, Sigma_x) + \
                               np.einsum('aj,ak,jk->ajk', nan_pattern_matrix_continuous, nan_pattern_matrix_continuous, np.identity(n_continuous))




        zero_padded_data = np.where(np.isnan(completed_data_continuous), 0, completed_data_continuous - mu_x.T - (W @ G.T @ completed_data_dummy.T).T)


        exponent_continuous = - 0.5 * np.sum(zero_padded_data * np.linalg.solve(Sigma_OO_array[nan_pattern_index_per_continuous, :, :], zero_padded_data.reshape((*zero_padded_data.shape, 1))).reshape(zero_padded_data.shape), axis=1)


        _, Sigma_log_det = np.linalg.slogdet(Sigma_OO_array)


        log_Z_continuous = (0.5 * Sigma_log_det + 0.5 * n_continuous_list * np.log(2*np.pi))[nan_pattern_index_per_continuous]

        log_Z_continuous = np.bincount(bincount_index_list, weights=log_Z_continuous) / np.bincount(bincount_index_list)




        log_joint_list = np.log(np.bincount(bincount_index_list, weights=np.exp(exponent_dummy + exponent_continuous)))


        log_joint_list = log_joint_list - log_Z_dummy - log_Z_continuous







        return log_joint_list






    def score(self, X, y=None):
        """
        Compute the average log-likelihood over the samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_mixture)
            Data with mixed continuous and binary variables for which the log-likelihood is computed.
            NaN values are allowed and treated as missing values.
            NumPy arrays and pandas DataFrames are supported.
        y : None, optional
            Ignored. The default is None.

        Returns
        -------
        float
            Average log-likelihood over the samples under the current model.

        """

        return float(np.mean(self.score_samples(X)))








    def bic(self):

        check_is_fitted(self, attributes='loglik_')

        if self.norm_constraint == True:
            bic = - 2 * self.loglik_ + (self.p_list_.size - self.n_continuous - self.n_dummy - self.n_components*(self.n_components - 1)*0.5 ) * np.log(self.n_samples_)
        else:
            bic = - 2 * self.loglik_ + (self.p_list_.size - self.n_components*(self.n_components - 1)*0.5 ) * np.log(self.n_samples_)


        return bic









    def sample(self, n_samples=1, random_state=None):
        """
        Generate random samples from the FactorAnalysis model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. The default is 1.
        random_state : int or None, optional
            Random seed for random sampling. The default is None.

        Returns
        -------
        generated_data : numpy.ndarray of shape (n_samples, n_mixture)
            Generated data.

        """



        p_list = self.p_list_
        n_components = self.n_components
        level_list = self.level_list
        norm_constraint = self.norm_constraint

        random_data = _sample_joint(p_list, level_list, n_components, n_samples, random_state=random_state, norm_constraint=norm_constraint)


        return random_data













    def get_mean_corr(self):

        level_list = self.level_list
        n_components = self.n_components
        norm_constraint = self.norm_constraint


        p_list = self.p_list_

        model_mean, rho_model = _fa_get_mean_corr(p_list, level_list, n_components, norm_constraint)


        return model_mean, rho_model






    def get_mean(self):
        """
        Compute the mean with the current model.

        Returns
        -------
        model_mean : numpy.ndarray of shape (n_mixture, )
            The mean of each feature under the current model.

        """

        model_mean, rho_model = self.get_mean_corr()

        return model_mean






    def get_corr(self):
        """
        Compute the correlation with the current model.

        Returns
        -------
        rho_model : numpy.ndarray of shape (n_mixture, n_mixture)
            The correlation among features under the current model.

        """


        model_mean, rho_model = self.get_mean_corr()



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


        model_mean, rho_model = self.get_mean_corr()


        fig = plt.figure(dpi=350)
        ax = fig.add_subplot(111)

        sns.heatmap(rho_model, annot=True, fmt='.2f', cmap='jet', vmin=-0.7, vmax=0.7, ax=ax)



        return ax










    def biplot(self, X, color_data=np.array([0]), pc_x=0, pc_y=1, marker_scale=1, arrow_scale=1, ax=None):



        import matplotlib.pyplot as plt



        level_list = self.level_list
        continuous_bool = self.continuous_bool
        dummy_bool = np.logical_not(continuous_bool)
        n_continuous = self.n_continuous
        n_dummy = self.n_dummy
        n_components = self.n_components
        n_mixture = n_continuous + n_dummy






        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")

        dummy_data = X[:, np.logical_not(continuous_bool)]

        valid_bool = np.isnan(dummy_data) | (dummy_data == 0) | (dummy_data == 1)
        if not np.all(valid_bool):
            raise ValueError("Dummy data must contain only 0, 1, or Nan.")









        contribution_ratio = self.contribution_ratio_

        factor_scores = self.transform(X)


        if np.sum(color_data) == 0:
            color_data = factor_scores[:, 0]



        X_unique, returned_index, order_index_per_original_data, unique_count_list = np.unique(X, return_index=True, return_inverse=True, return_counts=True, axis=0)

        # sort_index = np.argsort(returned_index)
        # X_unique = X_unique[sort_index, :]
        # unique_count_list = unique_count_list[sort_index]







        color_scores_unique = np.zeros(unique_count_list.size)
        factor_scores_unique = np.zeros((unique_count_list.size, n_components))
        for i_unique in range(unique_count_list.size):

            temp_bool = order_index_per_original_data == i_unique
            temp_index, = np.where(temp_bool)

            color_scores_unique[i_unique] = np.sum(color_data[temp_bool]) / np.sum(temp_bool)

            factor_scores_unique[i_unique, :] = factor_scores[temp_index[0], :]





        if ax == None:
            fig = plt.figure(figsize=(4, 4), dpi=350)
            ax = fig.add_subplot(111)




        _plot_scores(ax, factor_scores_unique, color_scores_unique, unique_count_list, contribution_ratio, title_text=None, pc_x=pc_x, pc_y=pc_y,
                    alpha_value=1, marker_scale=marker_scale, cmap='jet')








        W = self.continuous_components_
        G = self.dummy_components_

        Psi = np.diag(self.noise_variance_)


        feature_names = getattr(self, "feature_names_in_", list(np.arange(X.shape[1]).astype(str)))


        plot_G = G
        plot_W = np.diag(1/np.sqrt(np.diag(Psi))) @ W


        plot_M = np.zeros((n_mixture, n_components))
        plot_M[continuous_bool, :] = plot_W
        plot_M[dummy_bool, :] = plot_G




        y_min, y_max = ax.get_ylim()
        ylim_width = y_max - y_min

        temp_norm_max = np.max(np.sqrt(np.sum(plot_M[:, [pc_x, pc_y]]**2, axis=1)))
        temp_scale = 1/4 * ylim_width / temp_norm_max

        plot_M = plot_M * temp_scale



        _plot_arrows(ax, plot_M, level_list, feature_names, pc_x=pc_x, pc_y=pc_y, arrow_scale=arrow_scale)





        return ax











def _plot_scores(ax, factor_scores_unique, color_scores_unique, unique_count_list, contribution_ratio, title_text=None, pc_x=0, pc_y=1,
                alpha_value=1, marker_scale=1, cmap='jet', title_size=12, label_size=10, tick_size=8):



    N = np.sum(unique_count_list)

    if unique_count_list.size == N:
        marker_size_list = unique_count_list * 12
    elif unique_count_list.size > N / 2:
        marker_size_list = unique_count_list * 12
    else:
        marker_size_list = unique_count_list / np.max(unique_count_list) * 36 * 30





    temp_index = np.argsort(color_scores_unique)
    color_sorted = color_scores_unique[temp_index]
    unique_count_sorted = unique_count_list[temp_index]

    temp, = np.where((N * 1 / 4) < np.cumsum(unique_count_sorted))
    color_1st_quartile = color_sorted[temp[0]]

    temp, = np.where((N * 2 / 4) < np.cumsum(unique_count_sorted))
    color_median = color_sorted[temp[0]]

    temp, = np.where((N * 3 / 4) < np.cumsum(unique_count_sorted))
    color_3rd_quartile = color_sorted[temp[0]]



    color_vmax = np.nanmax(color_scores_unique)
    color_vmin = np.nanmin(color_scores_unique)






    bool_1 = (color_scores_unique < color_1st_quartile)
    bool_2 = (color_1st_quartile <= color_scores_unique) & (color_scores_unique < color_median)
    bool_3 = (color_median <= color_scores_unique) & (color_scores_unique < color_3rd_quartile)
    bool_4 = (color_3rd_quartile <= color_scores_unique)




    for temp_bool, marker in zip([bool_1, bool_2, bool_3, bool_4], ['o', 'o', 'o', 'o']):

        ax.scatter(factor_scores_unique[temp_bool, pc_x], factor_scores_unique[temp_bool, pc_y], alpha=alpha_value,
                   vmin=color_vmin, vmax=color_vmax, c=color_scores_unique[temp_bool],
                   s=marker_size_list[temp_bool]*marker_scale, cmap=cmap, marker=marker,
                   rasterized=True, zorder=1)



    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(0, color='black', linestyle='--')


    ax.tick_params(axis='both', labelsize=tick_size)


    ax.set_xlabel(f'dimension  {pc_x + 1} ({contribution_ratio[pc_x]*100:.1f}%)', fontsize=label_size, labelpad=4)
    ax.set_ylabel(f'dimension  {pc_y + 1} ({contribution_ratio[pc_y]*100:.1f}%)', fontsize=label_size, labelpad=4)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_axisbelow(True)



    ax.set_title(title_text, fontsize=title_size)


    return ax








def _plot_arrows(ax, plot_M, level_list, feature_names, pc_x=0, pc_y=1, arrow_scale=1, text_size=8):

    text_offset = 1.2



    plot_M = plot_M * arrow_scale



    import matplotlib.patches as patches

    circle = patches.Circle(xy=(0,0), radius=np.max(np.sqrt(np.sum(plot_M**2, axis=1))),
                            edgecolor='black', facecolor='none', linestyle='--', alpha=0.5)
    ax.add_patch(circle)





    from matplotlib.patheffects import withStroke
    effects = [withStroke(linewidth=2, foreground="white")]




    for i_loading, level in enumerate(level_list):


        temp_color = 'green' if level == 0 else 'red'



        plot_x, plot_y = plot_M[i_loading, pc_x], plot_M[i_loading, pc_y]

        ax.quiver(0, 0, plot_x, plot_y, angles='xy', scale_units='xy', scale=1, color=temp_color, zorder=2)



        text_x = plot_x * text_offset
        text_y = plot_y * text_offset



        ha = 'left' if text_x >= 0 else 'right'
        va = 'bottom' if text_y >= 0 else 'top'



        ax.text(x=text_x, y=text_y, s=feature_names[i_loading], color=temp_color, size=text_size, path_effects=effects, ha=ha, va=va)






    return ax















def _fa_get_mean_corr(p_list, level_list, n_components, norm_constraint=True):
    """
    Compute the mean and the correlation with the current model.

    Parameters
    ----------
    p_list : one-dimensional numpy.ndarray
        Optimizer-level model parameters.
    level_list : one-dimensional numpy.ndarray or list, dtype=int
        Level of measurement of each column in the data matrix. Each integer is encoded as follows:
        0 for a continuous variable, 1 for a binary variable.
    n_components : int
        Number of latent dimensions.
    norm_constraint : bool, optional
        Whether to impose the equal-norm constraint on the factor loading matrices. The default is True.

    Returns
    -------
    model_mean : numpy.ndarray of shape (n_mixture, )
        The mean of each feature under the current model.
    rho_model : numpy.ndarray of shape (n_mixture, n_mixture)
        The correlation among features under the current model.

    """




    continuous_bool = np.array([], dtype=np.bool_)
    for level in level_list:
        if level == 0:
            continuous_bool = np.append(continuous_bool, True)
        else:
            continuous_bool = np.append(continuous_bool, np.full(np.abs(level), False))

    dummy_bool = np.logical_not(continuous_bool)


    n_continuous = np.sum(continuous_bool)
    n_dummy = np.sum(dummy_bool)
    n_mixture = n_continuous + n_dummy



    if norm_constraint == True:
        mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_norm(p_list, n_continuous, n_dummy, n_components)
    else:
        mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_naive(p_list, n_continuous, n_dummy, n_components)


    mu_x = np.asarray(mu_x)
    Psi = np.asarray(Psi)
    W = np.asarray(W)
    b_list = np.asarray(b_list)
    G = np.asarray(G)
    c_z = np.asarray(c_z)



    Sigma_x = Psi + W @ W.T


    master_dummy = generate_master_dummy(level_list)





    if n_dummy != 0:

        from scipy.special import logsumexp

        Q_a = np.sum(master_dummy.T * ((np.diag(b_list) + 0.5 * G @ G.T) @ master_dummy.T), axis=0)


        log_pi_a = Q_a - logsumexp(Q_a)
        pi_a = np.exp(log_pi_a)


    else:
        pi_a = np.array([1])





    y_bar = np.einsum('a,aj->j', pi_a, master_dummy).reshape((-1,1))
    x_bar = mu_x + W @ G.T @ y_bar



    cov_yy = np.einsum('a,aj,ak->jk', pi_a, master_dummy, master_dummy) - y_bar @ y_bar.T
    rho_yy = np.einsum('jk,j,k->jk', cov_yy, 1/np.sqrt(np.diag(cov_yy)), 1/np.sqrt(np.diag(cov_yy)))


    cov_xx = Sigma_x + W @ G.T @ cov_yy @ G @ W.T
    rho_xx = np.einsum('jk,j,k->jk', cov_xx, 1/np.sqrt(np.diag(cov_xx)), 1/np.sqrt(np.diag(cov_xx)))


    cov_xy = W @ G.T @ cov_yy
    rho_xy = np.einsum('jk,j,k->jk', cov_xy, 1/np.sqrt(np.diag(cov_xx)), 1/np.sqrt(np.diag(cov_yy)))
    rho_yx = rho_xy.T






    rho_model = np.zeros((n_mixture, n_mixture))
    rho_model[np.ix_(continuous_bool, continuous_bool)] = rho_xx.copy()
    rho_model[np.ix_(continuous_bool, dummy_bool)] = rho_xy.copy()
    rho_model[np.ix_(dummy_bool, continuous_bool)] = rho_yx.copy()
    rho_model[np.ix_(dummy_bool, dummy_bool)] = rho_yy.copy()



    model_mean = np.zeros(n_mixture)
    model_mean[continuous_bool] = x_bar.ravel()
    model_mean[dummy_bool] = y_bar.ravel()




    return model_mean, rho_model












def _sample_joint(p_list, level_list, n_components, n_samples=1, random_state=None, norm_constraint=True):
    """
    Generate random samples from the FactorAnalysis model.

    Parameters
    ----------
    p_list : one-dimensional numpy.ndarray
        Optimizer-level model parameters.
    level_list : one-dimensional numpy.ndarray or list, dtype=int
        Level of measurement of each column in the data matrix. Each integer is encoded as follows:
        0 for a continuous variable, 1 for a binary variable.
    n_components : int
        Number of latent dimensions.
    n_samples : int, optional
        Number of samples to generate. The default is 1.
    random_state : int or None, optional
        Random seed for random sampling. The default is None.
    norm_constraint : bool, optional
        Whether to impose the equal-norm constraint on the factor loading matrices. The default is True.

    Returns
    -------
    generated_data : numpy.ndarray of shape (n_samples, n_mixture)
        Generated data.

    """




    rng = np.random.default_rng(random_state)


    continuous_bool = np.array([], dtype=np.bool_)
    for level in level_list:
        if level == 0:
            continuous_bool = np.append(continuous_bool, True)
        else:
            continuous_bool = np.append(continuous_bool, np.full(np.abs(level), False))

    dummy_bool = np.logical_not(continuous_bool)

    n_continuous = np.sum(continuous_bool)
    n_dummy = np.sum(dummy_bool)

    n_mixture = n_continuous + n_dummy


    if norm_constraint == True:
        mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_norm(p_list, n_continuous, n_dummy, n_components)
    else:
        mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_naive(p_list, n_continuous, n_dummy, n_components)


    mu_x = np.asarray(mu_x)
    Psi = np.asarray(Psi)
    W = np.asarray(W)
    b_list = np.asarray(b_list)
    G = np.asarray(G)
    c_z = np.asarray(c_z)



    Sigma_x = Psi + W @ W.T

    master_dummy = generate_master_dummy(level_list)




    generated_data = np.zeros((n_samples, n_mixture))

    if n_dummy > 0:

        from scipy.special import logsumexp

        b_a = np.einsum('j,ja->a', b_list, master_dummy.T)
        g_a = G.T @ master_dummy.T
        Q_a = b_a + 0.5 * np.sum(g_a**2, axis=0)

        pi_a = np.exp(Q_a - logsumexp(Q_a))



        randint_list = rng.choice(a=pi_a.size, size=n_samples, replace=True, p=pi_a)

        dummy_data = master_dummy[randint_list, :]

        generated_data[:, dummy_bool] = dummy_data

    else:
        dummy_data = np.empty((n_samples, 0))



    if n_continuous > 0:

        continuous_data = rng.multivariate_normal(mu_x.ravel(), Sigma_x, size=n_samples)

        continuous_data = continuous_data + (W @ G.T @ dummy_data.T).T

        generated_data[:, continuous_bool] = continuous_data


    return generated_data










def _rotate_loadings(p_list, level_list, n_components, norm_constraint=True):
    """
    Rotate latent space.

    Parameters
    ----------
    p_list : one-dimensional numpy.ndarray
        Optimizer-level model parameters.
    level_list : one-dimensional numpy.ndarray or list, dtype=int
        Level of measurement of each column in the data matrix. Each integer is encoded as follows:
        0 for a continuous variable, 1 for a binary variable.
    n_components : int
        Number of latent dimensions.
    norm_constraint : bool, optional
        Whether to impose the equal-norm constraint on the factor loading matrices. The default is True.

    Returns
    -------
    W : numpy.ndarray of shape (n_continuous, n_components)
        Factor loading matrix for continuous variables.
    G : numpy.ndarray of shape (n_dummy, n_components)
        Factor loading matrix for dummy variables.
    eigen_values : np.ndarray of shape (n_components, )
        Eigenvalues of combined dimensionless factor loading matrix M_tilde^T M_tilde.

    """



    continuous_bool = np.array([], dtype=np.bool_)
    for level in level_list:
        if level == 0:
            continuous_bool = np.append(continuous_bool, True)
        else:
            continuous_bool = np.append(continuous_bool, np.full(np.abs(level), False))

    n_continuous = np.sum(continuous_bool)
    n_dummy = np.sum(np.logical_not(continuous_bool))



    if norm_constraint == True:
        mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_norm(p_list, n_continuous, n_dummy, n_components)
    else:
        mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_naive(p_list, n_continuous, n_dummy, n_components)


    mu_x = np.asarray(mu_x)
    Psi = np.asarray(Psi)
    W = np.asarray(W)
    b_list = np.asarray(b_list)
    G = np.asarray(G)
    c_z = np.asarray(c_z)




    W_tilde = np.diag(1 / np.sqrt(np.diag(Psi))) @ W

    G_tilde = G




    M_tilde = np.vstack((W_tilde, G_tilde))
    eigen_values, R = np.linalg.eigh(M_tilde.T @ M_tilde)
    M_tilde = M_tilde @ R

    sort_index = np.argsort(eigen_values)[::-1]
    M_tilde = M_tilde[:, sort_index]
    eigen_values = eigen_values[sort_index]


    sign_list = np.sign(np.sum(M_tilde, axis=0))
    sign_list = np.where(sign_list == 0, 1, sign_list)
    M_tilde = M_tilde * sign_list.reshape((1,-1))


    W_tilde = M_tilde[:n_continuous, :].copy()
    G_tilde = M_tilde[n_continuous:, :].copy()


    W = np.diag(np.sqrt(np.diag(Psi))) @ W_tilde

    G =  G_tilde



    return W, G, eigen_values


















def _transform_data(p_list, level_list, n_components, X, norm_constraint=True):
    """
    Apply dimensionality reduction to the data using the model.

    Parameters
    ----------
    p_list : one-dimensional numpy.ndarray
        Optimizer-level model parameters.
    level_list : one-dimensional numpy.ndarray or list, dtype=int
        Level of measurement of each column in the data matrix. Each integer is encoded as follows:
        0 for a continuous variable, 1 for a binary variable.
    n_components : int
        Number of latent dimensions.
    X : array-like of shape (n_samples, n_mixture)
        Data with mixed continuous and binary variables, where binary variables are encoded as 0 or 1.
        NaN values are allowed and treated as missing values.
        NumPy arrays and pandas DataFrames are supported.
    norm_constraint : bool, optional
        Whether to impose the equal-norm constraint on the factor loading matrices. The default is True.

    Returns
    -------
    factor_scores : numpy.ndarray of shape (n_samples, n_components)
        Transformed array of factor scores.

    """


    continuous_bool = np.array([], dtype=np.bool_)
    for level in level_list:
        if level == 0:
            continuous_bool = np.append(continuous_bool, True)
        else:
            continuous_bool = np.append(continuous_bool, np.full(np.abs(level), False))

    dummy_bool = np.logical_not(continuous_bool)

    n_continuous = np.sum(continuous_bool)
    n_dummy = np.sum(dummy_bool)



    if norm_constraint == True:
        mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_norm(p_list, n_continuous, n_dummy, n_components)
    else:
        mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_naive(p_list, n_continuous, n_dummy, n_components)


    mu_x = np.asarray(mu_x)
    Psi = np.asarray(Psi)
    W = np.asarray(W)
    b_list = np.asarray(b_list)
    G = np.asarray(G)
    c_z = np.asarray(c_z)



    W, G, s_eigen_values = _rotate_loadings(p_list, level_list, n_components, norm_constraint)


    Sigma_x = Psi + W @ W.T


    master_dummy = generate_master_dummy(level_list)

    mu_z = - 1/2 * np.sum(G, axis=0)
    mu_z = mu_z.reshape((-1,1))










    completed_data, bincount_index_list = _complete_nan_data(X, level_list)



    completed_data_continuous = completed_data[:, continuous_bool]
    completed_data_dummy = completed_data[:, dummy_bool]




    nan_pattern_matrix_continuous, nan_pattern_index_per_continuous, nan_nnn_continuous = np.unique(np.isnan(completed_data_continuous), return_inverse=True, return_counts=True, axis=0)

    non_nan_pattern_matrix_continuous = np.logical_not(nan_pattern_matrix_continuous).astype(np.int_)



    Sigma_OO_array = np.einsum('aj,ak,jk->ajk', non_nan_pattern_matrix_continuous, non_nan_pattern_matrix_continuous, Sigma_x) + \
                            np.einsum('aj,ak,jk->ajk', nan_pattern_matrix_continuous, nan_pattern_matrix_continuous, np.identity(n_continuous))



    Sigma_ZO_array = np.einsum('aj,js->asj', non_nan_pattern_matrix_continuous, W)


    SSS = np.einsum('ajs->asj', np.linalg.solve(Sigma_OO_array, np.einsum('asj->ajs', Sigma_ZO_array)))



    zero_padded_data = np.where(np.isnan(completed_data_continuous), 0, completed_data_continuous - mu_x.T)


    W_nan = np.einsum('aj,js->ajs', non_nan_pattern_matrix_continuous, W)

    Sigma_Z_given_O_array = np.identity(n_components) - np.einsum('ajs,ajt->ast', W_nan, np.linalg.solve(Sigma_OO_array, W_nan))



    mu_a = mu_z.T + np.einsum('asj,aj->as', SSS[nan_pattern_index_per_continuous, :, :], zero_padded_data) \
        + np.einsum('ast,tj,ja->as', Sigma_Z_given_O_array[nan_pattern_index_per_continuous, :, :], G.T, completed_data_dummy.T)






    if n_dummy == 0:

        factor_scores = mu_a

    else:

        _, returned_index, master_dummy_index_per_completed_data, _ = np.unique(np.vstack((master_dummy, completed_data_dummy)), return_index=True, return_inverse=True, return_counts=True, axis=0)

        master_dummy_index_per_completed_data = returned_index[master_dummy_index_per_completed_data]

        master_dummy_index_per_completed_data = master_dummy_index_per_completed_data[master_dummy.shape[0]:]



        Q_a = np.einsum('j,aj->a', b_list, master_dummy)[master_dummy_index_per_completed_data] \
            + 0.5 * np.einsum('aj,ajk,ak->a', master_dummy[master_dummy_index_per_completed_data, :], np.einsum('js,ast,tk->ajk', G, Sigma_Z_given_O_array, G.T)[nan_pattern_index_per_continuous, :, :], master_dummy[master_dummy_index_per_completed_data, :]) \
            + np.einsum('aj,js,as->a', master_dummy[master_dummy_index_per_completed_data], G, np.einsum('asj,aj->as', SSS[nan_pattern_index_per_continuous, :, :], zero_padded_data))



        weighted_mu_a = np.einsum('a,as->as', np.exp(Q_a), mu_a)


        factor_scores = np.column_stack([np.bincount(bincount_index_list, weights=w) for w in weighted_mu_a.T]) / np.bincount(bincount_index_list, weights=np.exp(Q_a)).reshape((-1,1))






    return factor_scores































def _trial_iter(level_list, n_components, X, n_trial=1, tol=1e-12, maxiter=10000, random_state=None, norm_constraint=True):
    """
    Generate random initial parameter values and perform optimization for each random initialization.

    Parameters
    ----------
    level_list : one-dimensional numpy.ndarray or list, dtype=int
        Level of measurement of each column in the data matrix. Each integer is encoded as follows:
        0 for a continuous variable, 1 for a binary variable.
    n_components : int
        Number of latent dimensions.
    X : array-like of shape (n_samples, n_mixture)
        Data with mixed continuous and binary variables, where binary variables are encoded as 0 or 1.
        NaN values are allowed and treated as missing values.
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
    norm_constraint : bool, optional
        Whether to impose the equal-norm constraint on the factor loading matrices.
        If True, the row norms of the dimensionless factor loading matrices are constrained to be equal across all observed variables.
        If False, parameter estimation may yield improper solutions. The default is True.

    Raises
    ------
    RuntimeError
        If all optimization trials failed.

    Returns
    -------
    p_list : one-dimensional numpy.ndarray, dtype=float
        Best optimizer-level model parameters.
    status : int
        Termination status code of the optimizer for the best trial.
    fun : float
        Best average negative log-likelihood over the samples.
    n_iter : int
        Number of iterations performed by the optimizer for the best trial.
    elapsed_time : float
        Elapsed optimization time in seconds for the best trial.
    max_abs_grad : float
        Maximum absolute gradient of the average negative log-likelihood over the samples.
    n_hess_minus_eigvals : int
        Number of negative eigenvalues of a Hessian matrix.

    """





    master_dummy = generate_master_dummy(level_list)


    N = X.shape[0]
    N_inv = 1 / N



    continuous_bool = np.array([], dtype=np.bool_)
    for level in level_list:
        if level==0:
            continuous_bool = np.append(continuous_bool, True)
        else:
            continuous_bool = np.append(continuous_bool, np.full(np.abs(level), False))

    dummy_bool = np.logical_not(continuous_bool)

    n_continuous = np.sum(continuous_bool)
    n_dummy = np.sum(dummy_bool)


    dummy_data = X[:, dummy_bool]











    observed_data_1 = X[np.sum(np.isnan(X[:, dummy_bool]), axis=1) == 0, :]
    observed_data_1 = np.hstack((observed_data_1[:, continuous_bool], observed_data_1[:, dummy_bool]))
    observed_data_1_continuous = observed_data_1[:, :n_continuous]
    observed_data_1_dummy = observed_data_1[:, n_continuous:]


    if np.sum(dummy_bool) > 0:

        _, returned_index, _, nnn_observed = np.unique(np.vstack((master_dummy, observed_data_1_dummy)),
                                                    return_index=True, return_inverse=True, return_counts=True, axis=0)

        sort_index = np.argsort(returned_index)
        nnn_observed = nnn_observed[sort_index]
        qqq_1 = (nnn_observed - 1) * N_inv

    else:
        qqq_1 = np.array([])





    nan_pattern_matrix_continuous_1, target_index_per_observed_data_1, nan_nnn_continuous_1 = np.unique(np.isnan(observed_data_1_continuous),
                                                                                      return_inverse=True, return_counts=True, axis=0)


    index_bool_per_target_observed = np.zeros((nan_nnn_continuous_1.size, observed_data_1.shape[0]), dtype=np.int_)

    for i_order in range(nan_nnn_continuous_1.size):

        index_bool_per_target_observed[i_order, :] = target_index_per_observed_data_1 == i_order





    pad_data = np.where(np.isnan(observed_data_1), 0.0, observed_data_1)

    empirical_mean_array_1 = np.einsum('a,ai,ij->aj', 1/nan_nnn_continuous_1, index_bool_per_target_observed, pad_data)


    temp_diff = pad_data - np.einsum('iaj,ij->aij',
                                     np.broadcast_to(empirical_mean_array_1, (observed_data_1.shape[0], nan_nnn_continuous_1.size, observed_data_1.shape[1])),
                                     np.logical_not(np.isnan(observed_data_1)))


    empirical_cov_array_1 = np.einsum('a,ai,aij,aik->ajk', 1/nan_nnn_continuous_1, index_bool_per_target_observed, temp_diff, temp_diff)














    nan_data = X[np.sum(np.isnan(dummy_data), axis=1) != 0, :]

    completed_data, bincount_index_list = _complete_nan_data(nan_data, level_list)



    completed_data = np.hstack((completed_data[:, continuous_bool], completed_data[:, dummy_bool]))



    completed_data_continuous = completed_data[:, :n_continuous]
    completed_data_dummy = completed_data[:, n_continuous:]



    _, returned_index, master_dummy_index_per_completed_data, _ = np.unique(np.vstack((master_dummy, completed_data_dummy)), return_index=True, return_inverse=True, return_counts=True, axis=0)


    master_dummy_index_per_completed_data = returned_index[master_dummy_index_per_completed_data]

    master_dummy_index_per_completed_data = master_dummy_index_per_completed_data[master_dummy.shape[0]:]


    nan_pattern_matrix_continuous_2, nan_pattern_index_per_continuous_2, nan_nnn_continuous_2 = np.unique(np.isnan(completed_data_continuous), return_inverse=True, return_counts=True, axis=0)









    def callback(p_list):
        return _NLL(p_list,
                    n_continuous, n_dummy, master_dummy,
                    qqq_1, nan_pattern_matrix_continuous_1, nan_nnn_continuous_1, empirical_mean_array_1, empirical_cov_array_1,
                    completed_data, bincount_index_list, master_dummy_index_per_completed_data,
                    nan_pattern_matrix_continuous_2, nan_nnn_continuous_2, nan_pattern_index_per_continuous_2,
                    n_components, norm_constraint)



    rng = np.random.default_rng(random_state)


    if norm_constraint == True:
        p_list_size = _par_initial_norm(level_list, n_components).size
    else:
        p_list_size = _par_initial_naive(level_list, n_components).size



    result_list = np.array([]).reshape(0, 5)
    p_matrix = np.array([]).reshape(0, p_list_size)
    diagnose_list = np.array([]).reshape((0, 4))
    error_list = []

    for i_trial in range(n_trial):


        try:


            if norm_constraint == True:
                p_list = _par_initial_norm(level_list, n_components, rng=rng)
            else:
                p_list = _par_initial_naive(level_list, n_components, rng=rng)




            n_iter = 0

            ti_single_trial = time.perf_counter()




            result = optimize.minimize(_NLL_and_grad, x0=p_list,
                                       args=(n_continuous, n_dummy, master_dummy,
                                             qqq_1, nan_pattern_matrix_continuous_1, nan_nnn_continuous_1, empirical_mean_array_1, empirical_cov_array_1,
                                             completed_data, bincount_index_list, master_dummy_index_per_completed_data,
                                             nan_pattern_matrix_continuous_2, nan_nnn_continuous_2, nan_pattern_index_per_continuous_2,
                                             n_components, norm_constraint),
                                        jac=True,
                                        method='BFGS',
                                        options={'gtol': tol, 'maxiter': maxiter, 'disp': False} )


            p_list = result.x.copy()

            n_iter = n_iter + result.nit




            tf_single_trial = time.perf_counter()




            if result.fun > - 9999:


                result_list = np.vstack((result_list, np.array([result.success,
                                                                result.status,
                                                                result.fun,
                                                                n_iter,
                                                                tf_single_trial - ti_single_trial])))

                p_matrix = np.vstack((p_matrix, result.x))





                hessian_matrix = jax.hessian(callback)(p_list)
                hessian_matrix = np.asarray(hessian_matrix)


                eigvals = np.linalg.eigvalsh(hessian_matrix)

                diagnose_list = np.vstack((diagnose_list,
                                           np.array([np.max(np.abs(result.jac)),
                                                     np.min(eigvals),
                                                     np.max(eigvals),
                                                     np.sum(eigvals < -1e-8),]),))





        except (ValueError, np.linalg.LinAlgError) as error:
            error_list.append(error)






    else:
        if result_list.shape[0] == 0:
            raise RuntimeError("All optimization trials failed.")









    temp_bool = (result_list[:, 1] == 0) | (result_list[:, 1] == 9) | (result_list[:, 1] == 2)
    result_list = result_list[temp_bool, :]
    p_matrix = p_matrix[temp_bool, :]
    diagnose_list = diagnose_list[temp_bool, :]


    if result_list.shape[0] == 0:
        raise RuntimeError("Optimization failed.")




    sort_index = np.lexsort((diagnose_list[:, 0], np.round(result_list[:, 2], 5), result_list[:, 1]))
    result_list = result_list[sort_index, :]
    p_matrix = p_matrix[sort_index, :]
    diagnose_list = diagnose_list[sort_index, :]






    p_list = p_matrix[0, :].copy()
    success = result_list[0, 0]
    status = result_list[0, 1]
    fun = result_list[0, 2]
    n_iter = result_list[0, 3]
    elapsed_time = result_list[0, 4]


    max_abs_grad = diagnose_list[0, 0]
    n_hess_minus_eigvals = diagnose_list[0, 3]



    if success == False:
        import warnings
        warnings.warn("Optimization may not have converged."
                      "Check the gradient norm.",
                      RuntimeWarning,
                      stacklevel=2
                      )




    return p_list, int(status), float(fun), int(n_iter), float(elapsed_time), float(max_abs_grad), int(n_hess_minus_eigvals)




























def _par_initial_norm(level_list, n_components, rng=None):
    """
    Generate initial model parameters randomly.

    Parameters
    ----------
    level_list : one-dimensional numpy.ndarray or list, dtype=int
        Level of measurement of each column in the data matrix. Each integer is encoded as follows:
        0 for a continuous variable, 1 for a binary variable.
    n_components : int
        Number of latent dimensions.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, a new generator is created using ``numpy.random.default_rng()``.
        The default is None.

    Returns
    -------
    p_list : one-dimensional numpy.ndarray, dtype=float
        Optimizer-level model parameters.

    """


    continuous_bool = np.array([], dtype=np.bool_)
    for level in level_list:
        if level == 0:
            continuous_bool = np.append(continuous_bool, True)
        else:
            continuous_bool = np.append(continuous_bool, np.full(np.abs(level), False))

    dummy_bool = np.logical_not(continuous_bool)

    n_continuous = np.sum(continuous_bool)
    n_dummy = np.sum(dummy_bool)



    if rng == None:
        rng = np.random.default_rng()


    p_list = rng.normal(loc=0, scale=1/2, size=n_dummy + 2*n_continuous + (n_dummy + n_continuous)*n_components + 1)




    return p_list





def _reconstruct_parameters_norm(p_list, n_continuous, n_dummy, n_components):
    """
    Construct the parameters of factor analysis from optimizer-level model parameters.

    Parameters
    ----------
    p_list : one-dimensional numpy.ndarray
        Optimizer-level model parameters.
    n_continuous : int
        Number of continuous variables.
    n_dummy : int
        Number of binary variables.
    n_components : int
        Number of latent dimensions.

    Returns
    -------
    mu_x : jax.Array of shape (n_continuous, 1)
        Mean vector for continuous variables.
    Psi : jax.Array of shape (n_continuous, n_continuous)
        Diagonal matrix of unique variances.
    W : jax.Array of shape (n_continuous, n_components)
        Factor loading matrix for continuous variables.
    b_list : jax.Array of shape (n_dummy, )
        Bias parameters for dummy variables.
    G : jax.Array of shape (n_dummy, n_components)
        Factor loading matrix for dummy variables.
    c_z : jax.Array of shape (n_mixture, )
        Row norms of the dimensionless factor loading matrices Psi^{-1/2} W and G.

    """



    b_list = p_list[0:n_dummy]
    g_raw_list = p_list[n_dummy:(n_dummy*n_components + n_dummy)]


    mu_x_list = p_list[(n_dummy*n_components + n_dummy):(n_dummy*n_components + n_dummy + n_continuous)]
    psi_log_list = p_list[(n_dummy*n_components + n_dummy + n_continuous):(n_dummy*n_components + n_dummy + 2*n_continuous)]
    W_raw = p_list[(n_dummy*n_components + n_dummy + 2*n_continuous):(-1)].reshape((n_continuous, n_components))



    c_z_log_list = p_list[-1]
    c_z = jnp.full(n_continuous + n_dummy, jnp.exp(jnp.clip(c_z_log_list, None, 100)))



    G_raw = g_raw_list.reshape((n_dummy, n_components))

    G_raw_norm = jnp.sqrt(jnp.sum(G_raw**2, axis=1))
    G_hat = G_raw / G_raw_norm.reshape((n_dummy, 1))


    G = G_hat * c_z[0]



    W_raw_norm = jnp.sqrt(jnp.sum(W_raw**2, axis=1))
    W_hat = W_raw / W_raw_norm.reshape((n_continuous, 1))

    W = jnp.diag(jnp.exp(1/2*jnp.clip(psi_log_list, None, 100))) @ W_hat * c_z[0]



    Psi = jnp.diag(jnp.exp(jnp.clip(psi_log_list, None, 100)))

    mu_x = mu_x_list.reshape((n_continuous, 1))



    return mu_x, Psi, W, b_list, G, c_z






















def _par_initial_naive(level_list, n_components, rng=None):



    continuous_bool = np.array([], dtype=np.bool_)
    for i_feature in range(len(level_list)):
        temp_level = np.abs(level_list[i_feature])
        if temp_level==0:
            continuous_bool = np.append(continuous_bool, True)
        else:
            continuous_bool = np.append(continuous_bool, np.full(temp_level, False))

    n_continuous = np.sum(continuous_bool)
    n_dummy = np.sum(np.logical_not(continuous_bool))



    if rng == None:
        rng = np.random.default_rng()


    p_list = rng.normal(loc=0, scale=1/2, size=n_dummy + 2*n_continuous + (n_dummy + n_continuous)*n_components)




    return p_list





def _reconstruct_parameters_naive(p_list, n_continuous, n_dummy, n_components):


    b_list = p_list[0:n_dummy]
    g_list = p_list[n_dummy:(n_dummy*n_components + n_dummy)]


    mu_x_list = p_list[(n_dummy*n_components + n_dummy):(n_dummy*n_components + n_dummy + n_continuous)]
    psi_log_list = p_list[(n_dummy*n_components + n_dummy + n_continuous):(n_dummy*n_components + n_dummy + 2*n_continuous)]
    W_tilde = p_list[(n_dummy*n_components + n_dummy + 2*n_continuous):].reshape((n_continuous, n_components))


    G = g_list.reshape((n_dummy, n_components))



    W = jnp.diag(jnp.exp(1/2*jnp.clip(psi_log_list, None, 100))) @ W_tilde

    Psi = jnp.diag(jnp.exp(jnp.clip(psi_log_list, None, 100)))


    mu_x = mu_x_list.reshape((n_continuous, 1))


    c_z = jnp.append(jnp.sqrt(jnp.sum(W_tilde**2, axis=1)),
                     jnp.sqrt(jnp.sum(G**2, axis=1)))


    return mu_x, Psi, W, b_list, G, c_z


























import jax.numpy as jnp
import jax
import jax.scipy.special



def _NLL(p_list, n_continuous, n_dummy, master_dummy,
        qqq_1, nan_pattern_matrix_continuous_1, nan_nnn_continuous_1, empirical_mean_array_1, empirical_cov_array_1,
        completed_data, bincount_index_list, master_dummy_index_per_completed_data,
        nan_pattern_matrix_continuous_2, nan_nnn_continuous_2, nan_pattern_index_per_continuous_2,
        n_components, norm_constraint=True):


    if norm_constraint == True:
        mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_norm(p_list, n_continuous, n_dummy, n_components)
    else:
        mu_x, Psi, W, b_list, G, c_z = _reconstruct_parameters_naive(p_list, n_continuous, n_dummy, n_components)




    Sigma_x = Psi + W @ W.T



    if n_dummy != 0:
        Q_a = jnp.sum(master_dummy.T * ((jnp.diag(b_list) + 0.5 * G @ G.T) @ master_dummy.T), axis=0)

        log_Z_binary = jax.scipy.special.logsumexp(Q_a)

    else:
        Q_a = jnp.array([])
        log_Z_binary = 0



    N_inv = 1 / jnp.sum(nan_nnn_continuous_1) if bincount_index_list.size==0 else 1 / (jnp.sum(nan_nnn_continuous_1) + jnp.max(bincount_index_list) + 1)



    psi_log_list = jnp.log(jnp.diag(Psi))
    Psi_inv = jnp.diag(jnp.exp(- psi_log_list))
    W_tilde = jnp.exp(-1/2 * psi_log_list).reshape((-1,1)) * W








    # data_1
    l_binary_1 = - jnp.sum(qqq_1 * Q_a)


    non_nan_pattern_matrix_continuous_1 = jnp.logical_not(nan_pattern_matrix_continuous_1).astype(jnp.int_)

    n_continuous_list_1 = jnp.sum(non_nan_pattern_matrix_continuous_1, axis=1)




    W_nan = jnp.einsum('aj,js->ajs', non_nan_pattern_matrix_continuous_1, W)



    x_mean_diff = empirical_mean_array_1[:, :n_continuous] - jnp.einsum('aj,j->aj', non_nan_pattern_matrix_continuous_1, mu_x.ravel()) \
                  - jnp.einsum('ajs,sk,ak->aj', W_nan, G.T, empirical_mean_array_1[:, n_continuous:])

    S_array = empirical_cov_array_1[:, :n_continuous, :n_continuous] \
                            - jnp.einsum('ajs,sl,alk->ajk', W_nan, G.T, empirical_cov_array_1[:, n_continuous:, :n_continuous]) \
                            - jnp.einsum('ajl,ls,aks->ajk', empirical_cov_array_1[:, :n_continuous, n_continuous:], G, W_nan) \
                            + jnp.einsum('ajs,sl,alm,mt,akt->ajk', W_nan, G.T, empirical_cov_array_1[:, n_continuous:, n_continuous:], G, W_nan) \
                            + jnp.einsum('aj,ak->ajk', x_mean_diff, x_mean_diff)



    WWW_nan = jnp.einsum('aj,js->ajs', non_nan_pattern_matrix_continuous_1, Psi_inv @ W)


    woodbury_matrix_array = jnp.broadcast_to(jnp.identity(n_components), (nan_nnn_continuous_1.size, n_components, n_components)) + \
                            jnp.einsum('sj,aj,jt->ast', W_tilde.T, non_nan_pattern_matrix_continuous_1, W_tilde)


    Sigma_inv_array = jnp.broadcast_to(jnp.identity(n_continuous), (nan_nnn_continuous_1.size, n_continuous, n_continuous)) * Psi_inv.reshape((1, n_continuous, n_continuous)) - \
                            jnp.einsum('ajs,ask->ajk', WWW_nan, jnp.linalg.solve(woodbury_matrix_array, jnp.einsum('ajs->asj', WWW_nan)))



    l_continuous_list = (nan_nnn_continuous_1 * N_inv) * 1/2 * jnp.einsum('ajk,akl,jl->a', Sigma_inv_array, S_array, jnp.identity(n_continuous))


    l_continuous_1 = jnp.sum(l_continuous_list)






    Sigma_array = jnp.einsum('aj,ak,jk->ajk', non_nan_pattern_matrix_continuous_1, non_nan_pattern_matrix_continuous_1, Sigma_x) + \
                           jnp.einsum('aj,ak,jk->ajk', nan_pattern_matrix_continuous_1, nan_pattern_matrix_continuous_1, jnp.identity(n_continuous))


    L_array = jnp.linalg.cholesky(Sigma_array)


    log_Z_continuous_list = (n_continuous_list_1 / 2) * jnp.log(2 * jnp.pi)  + \
                             jnp.sum(jnp.log(jnp.diagonal(L_array, axis1=1, axis2=2)), axis=1)

    log_Z_continuous_1 = jnp.sum(nan_nnn_continuous_1 * log_Z_continuous_list) * N_inv













    # data_2
    if completed_data.shape[0] != 0:

        completed_data_continuous = completed_data[:, :n_continuous]
        completed_data_dummy = completed_data[:, n_continuous:]



        exponent_dummy = Q_a[master_dummy_index_per_completed_data]



        temp_diff = jnp.where(jnp.isnan(completed_data_continuous.T), 0.0, completed_data_continuous.T - mu_x - W @ G.T @ completed_data_dummy.T)

        non_nan_pattern_matrix_continuous_2 = jnp.logical_not(nan_pattern_matrix_continuous_2).astype(jnp.int_)




        W_tilde_nan = jnp.einsum('aj,js->ajs', non_nan_pattern_matrix_continuous_2, W_tilde)

        woodbury_matrix_array = jnp.broadcast_to(jnp.identity(n_components), (nan_pattern_matrix_continuous_2.shape[0], n_components, n_components)) + \
                                      jnp.einsum('sj,aj,jt->ast', W_tilde.T, non_nan_pattern_matrix_continuous_2, W_tilde)


        Sigma_inv_nan_unique = jnp.tile(Psi_inv, (nan_pattern_matrix_continuous_2.shape[0], 1, 1)) - \
            jnp.einsum('j,ajs,ask->ajk', jnp.exp(- 1/2 * psi_log_list), W_tilde_nan,
                        jnp.linalg.solve(woodbury_matrix_array, jnp.einsum('ajs,j->asj', W_tilde_nan, jnp.exp(- 1/2 * psi_log_list))))


        exponent_continuous = - 0.5 * jnp.einsum('ji,ijk,ki->i', temp_diff, Sigma_inv_nan_unique[nan_pattern_index_per_continuous_2, :, :], temp_diff)


        log_prob_list_completed_data = exponent_dummy + exponent_continuous

        log_prob_list = jnp.log(jnp.bincount(bincount_index_list, weights=jnp.exp(log_prob_list_completed_data)))




        n_continuous_list_2 = jnp.sum(non_nan_pattern_matrix_continuous_2, axis=1)


        Sigma_array = jnp.einsum('aj,ak,jk->ajk', non_nan_pattern_matrix_continuous_2, non_nan_pattern_matrix_continuous_2, Sigma_x) + \
                               jnp.einsum('aj,ak,jk->ajk', nan_pattern_matrix_continuous_2, nan_pattern_matrix_continuous_2, jnp.identity(n_continuous))


        L_array = jnp.linalg.cholesky(Sigma_array)


        log_Z_continuous_list = (n_continuous_list_2 / 2) * jnp.log(2 * jnp.pi)  + \
                                 jnp.sum(jnp.log(jnp.diagonal(L_array, axis1=1, axis2=2)), axis=1)



        log_Z_continuous_2 = jnp.bincount(bincount_index_list, weights=log_Z_continuous_list[nan_pattern_index_per_continuous_2]) / jnp.bincount(bincount_index_list)


        log_Z_continuous_2 = jnp.sum(log_Z_continuous_2) * N_inv


    else:

        log_prob_list = jnp.array([])
        log_Z_continuous_2 = 0




    fval = log_Z_binary + log_Z_continuous_1 + l_binary_1 + l_continuous_1 - N_inv * jnp.sum(log_prob_list) + log_Z_continuous_2




    return fval






from jax import value_and_grad, jit

_NLL_and_grad = jit(value_and_grad(_NLL), static_argnames=['n_continuous', 'n_dummy', 'n_components', 'norm_constraint', ])





def _complete_nan_data(nan_data, level_list):
    """
    Fill in the missing values in the binary variables with all possible patterns of dummy variables.

    Parameters
    ----------
    nan_data : array-like of shape (n_samples, n_mixture)
        Data with mixed continuous and binary variables, where binary variables are encoded as 0 or 1.
        Missing values are represented by NaN.
        NumPy arrays and pandas DataFrames are supported.
    level_list : one-dimensional numpy.ndarray or list, dtype=int
        Level of measurement of each column in the data matrix. Each integer is encoded as follows:
        0 for a continuous variable, 1 for a binary variable.

    Returns
    -------
    completed_data : numpy.ndarray of shape (completed_data.shape[0], n_mixture)
        Data with the dummy variables completeed.
    bincount_index_list : one-dimensional numpy.ndarray of shape (completed_data.shape[0], ), dtype=int
        An integer index array indicating which nan_data points correspond to each completed_data.

    """

    level_list = np.array(level_list, dtype=np.int_)

    continuous_bool = np.array([], dtype=np.bool_)
    for level in level_list:
        if level == 0:
            continuous_bool = np.append(continuous_bool, True)
        else:
            continuous_bool = np.append(continuous_bool, np.full(np.abs(level), False))

    dummy_bool = np.logical_not(continuous_bool)







    completed_data = nan_data.copy()

    dummy_nan_count = np.sum(np.isnan(completed_data[:, dummy_bool]), axis=1)

    nan_index_list, = np.where(dummy_nan_count > 0)

    bincount_index_list = np.arange(nan_data.shape[0])

    tiled_N_cumulative = 0
    for nan_index in nan_index_list:


        temp_completed_data = nan_data[nan_index:(nan_index + 1), :]

        temp_dummy_nan_count = np.sum(np.isnan(temp_completed_data[:, dummy_bool]), axis=1)

        temp_nan_index_list, = np.where(temp_dummy_nan_count > 0)



        while len(temp_nan_index_list) > 0:


            temp_nan_index = temp_nan_index_list[0]



            temp_col = 0
            for temp_level in level_list:

                if temp_level == 0:
                    temp_col += 1
                    continue


                temp_data = temp_completed_data[temp_nan_index, temp_col:(temp_col + np.abs(temp_level))]

                if np.sum(np.isnan(temp_data)) == 0:
                    temp_col += np.abs(temp_level)
                    continue



                tiled_data = np.tile(temp_completed_data[temp_nan_index, :], (np.abs(temp_level) + 1, 1))


                tiled_data[:, temp_col:(temp_col + np.abs(temp_level))] = 0

                for i_level in range(1, np.abs(temp_level) + 1):

                    if temp_level > 0:

                        tiled_data[i_level, temp_col + i_level - 1] = 1

                    else:

                        tiled_data[i_level, temp_col:(temp_col + i_level)] = 1





                nan_bool = np.isnan(temp_data)
                if np.sum(nan_bool) != temp_data.size:

                    non_nan_bool = np.logical_not(np.isnan(temp_data))

                    temp_tiled_data = tiled_data[:, temp_col:(temp_col + np.abs(temp_level))]

                    completed_bool = np.all(temp_tiled_data[:, non_nan_bool] == temp_data[non_nan_bool], axis=1)
                    tiled_data = tiled_data[completed_bool, :]






                temp_completed_data = np.vstack((temp_completed_data[:temp_nan_index, :],
                                                 tiled_data,
                                                 temp_completed_data[(temp_nan_index + 1):, :]))



                break




            temp_dummy_nan_count = np.sum(np.isnan(temp_completed_data[:, dummy_bool]), axis=1)

            temp_nan_index_list, = np.where(temp_dummy_nan_count > 0)




        tiled_N = temp_completed_data.shape[0]

        insert_index = nan_index + tiled_N_cumulative

        completed_data = np.vstack((completed_data[:insert_index, :],
                                    temp_completed_data,
                                    completed_data[(insert_index + 1):, :]))


        bincount_index_list = np.concatenate([bincount_index_list[:insert_index],
                                              np.ones(tiled_N, dtype=np.int_) * nan_index,
                                              bincount_index_list[(insert_index + 1):]])


        tiled_N_cumulative = tiled_N_cumulative + (tiled_N - 1)





    return completed_data, bincount_index_list
