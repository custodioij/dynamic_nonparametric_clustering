"""
Code used in:
    Igor Custodio João, Julia Schaumburg, André Lucas, Bernd Schwaab, Dynamic Nonparametric Clustering of Multivariate Panel Data, Journal of Financial Econometrics, 2022.
    https://doi.org/10.1093/jjfinec/nbac038

This module contains a function for k-means clustering, and a class with models to perform Algorithm 2.

Typical usage might be:
    model = el.CLModel(data, ident)
    H = model.elastic(epsilon=epsilon)
    
See Simulate_clustering.py and Simulation_plots.py for more information on how to read the output H.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment  # For Hungarian Algorithm
from sklearn.metrics import confusion_matrix  # Build the confusion matrix
from sklearn.mixture import GaussianMixture  # To use gaussian mixture to cluster the cross-section
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics.pairwise import paired_distances
from scipy.cluster import hierarchy  # for the fcluster function
from scipy import stats
from scipy.stats import multivariate_normal
from collections import Counter
import itertools
import warnings


class CLModel(object):
    """
    Model and clustering functions.

    Uses the average silhouette and k-means to cluster.

    Attributes:
        N (int): number of points.
        T (int): number of cross-sections
        ident (list): list of lists containing the i and t of each point.
            E.g.: if N=2 and T=3, then ident = [[0,0], [0,1], [0,2],
                                                [1,0], [1,1], [1,2]]
        CS_l (list): list of matrices containing the cross-sections.
        unrealistic_centers (np.array): matrix containing the point to be used to fill the matrix of centers
        k_range (tuple): numbers of clusters to try.
        centers (list): list of matrices of centers for each cross-section.
        H (np.array): matrix containing all assignments, with one point per row.
        switches (np.array): matrix of same shape as H, with 1s for switches. First column is zero.
    """
    def __init__(self, data, ident=None, k_range=(2, 3, 4, 5)):
        self.N = None
        self.T = None
        self.ident = ident
        if ident is not None:
            self.N = max([i[0] for i in ident]) + 1
            self.T = max([i[1] for i in ident]) + 1
        self.CS_l = [i for i in self._generate_cs(data)]  # List of arrays.
        self.unrealistic_centers = 10 * np.amax(data, 0)  # An auxiliary array.
        self.k_range = k_range
        self.H = np.full((self.N, self.T), None)  # Matrix of assignments (empty).
        self.centers = [None for _ in range(self.T)]
        self.switches = None  # Built by the build_switches function, called by elastic
        self.confusion_tracker = []  # Used to track the cluster mappings across time
        self.taxonomy_tracker = []  # Used to track the cluster mappings across time
        self.silhouette_stat = []
        self.gini_weighted_silhouette_stat = []
        self.distance_to_cluster_mean = np.full((self.N, self.T), None)

    def _generate_cs(self, data, time=None):
        """ Intended to be a generator of cross-sections to use in functions such as kmeans. """
        if time is None:
            time = range(self.T)
        for t in time:
            indices = [i for i in range(data.shape[0]) if self.ident[i][1] == t]
            yield data[indices, :].copy()

    @staticmethod
    def assignment(positions, centers):
        """
        Assigns points to their nearest center.

        Args:
            positions (np.array): Matrix of a point in each row.
            centers (np.array): Matrix of centers, one center in each column.

        Returns:
            List of indices to which each point was assigned, correspomnding to the columns of centers.
        """
        centers_dist = cdist(positions, centers.T)
        cluster_assignments = [np.argmin(centers_dist[i, :]) for i in range(centers_dist.shape[0])]
        return cluster_assignments

    @staticmethod
    def silhouette(CS, centers, h=None, actual_assignment=False):
        """
        Calculates the average silhouette statistic for a clustered cross-section. Not defined for k = 1.
        Does its own assignment.
        actual_assignment uses the actual assignment. When FALSE is used to compute the closest from the positions

        Args:
            CS (np.array): Cross-section of points (n rows and p columns).
            centers (np.array): Matrix of centers, one cluster in each column.

        Returns:
            Average silhouette given the clustering.
        """
        m_dist = squareform(pdist(CS))
        if h is None:
            # Calculate the first and second closest clusters for each point
            centers_dist = cdist(CS, centers.T)
            first_closest = [np.argpartition(centers_dist[i, :], 0)[0] for
                             i in range(centers_dist.shape[0])]
            second_closest = [np.argpartition(centers_dist[i, :], 1)[1] for
                              i in range(centers_dist.shape[0])]
        else:
            centers = np.vstack([CS[[hh == i for hh in h], :].mean(axis=0) for i in set(h)]).T
            centers_dist = cdist(CS, centers.T)
            if actual_assignment:
                first_closest = list(h)
                for i in range(centers_dist.shape[0]):
                    centers_dist[i, [num for num, j in enumerate(set(h)) if j == h[i]]] = np.inf
                second_closest = [list(set(h))[np.argpartition(centers_dist[i, :], 0)[0]] for
                                  i in range(centers_dist.shape[0])]
            else:
                first_closest = [np.argpartition(centers_dist[i, :], 0)[0] for
                                 i in range(centers_dist.shape[0])]
                for i in range(centers_dist.shape[0]):
                    centers_dist[i, [num for num, j in enumerate(set(first_closest)) if
                                     j == first_closest[i]]] = np.inf
                for i in range(centers_dist.shape[1]):
                    if i not in set(first_closest):
                        centers_dist[:, i] = np.inf  # remove centers that are "empty"
                second_closest = [np.argpartition(centers_dist[i, :], 0)[0] for
                                  i in range(centers_dist.shape[0])]

        # Average distances
        a = [np.mean(m_dist[i, [j == first_closest[i] for j in first_closest]]) for
             i in range(centers_dist.shape[0])]
        b = [np.mean(m_dist[i, [j == second_closest[i] for j in first_closest]]) for
             i in range(centers_dist.shape[0])]

        s = []
        for i in range(centers_dist.shape[0]):
            s += [(b[i] - a[i])/(np.max([a[i], b[i]]))]
        return np.mean(s)


    def cs_clustering(self, t, k=None, CSmodel='kmeans', prev_clust=None, covar='tied'):
        """
        Cluster a cross-section for many values of k, calculates the average silhouette of each,
         and returns the optimal.

        Args:
            t (int): index of the cross-section to consider

        Returns:
            list of 4: centers array, the list of assignments,
                        and the optimal k.
        """
        if CSmodel == 'kmeans':
            CSfct = lambda x: kmeans(self.CS_l[t], k=x)
        if CSmodel == 'initialization':  # Ignores k.
            assert max(self.initial_clustering) <= k-1,\
                "Initial clustering has more clusters than it should"
            CSfct = lambda x: [np.array([np.mean(self.CS_l[t][np.array(self.initial_clustering) == i, :], axis=0) for
                                         i in range(max(self.initial_clustering)+1)]).T,
                               self.initial_clustering, max(self.initial_clustering)+1]
        assert k is not None
        fitted_clustering = list(CSfct(k))
        if CSmodel == 'kmeans':
            fitted_clustering += [k]
        return fitted_clustering

    def elastic(self, epsilon=0.2, gini_s=False, initial_clustering=None):
        """
        Performs the elastic clustering with varying k (algorithm 2),
         determining k after the relaxed assignment.

        Note:
            Dimension of X bar might not match the pseudo-code.

        Args:
            epsilon (float): elasticity parameter
            gini_s (bool): Whether or not to adjust the silhouette by the Gini index.
            initial_clustering (list): alternative initial cluster assignments.

        Returns:
            Matrix of assignments H.
        """
        MSE = []
        first_t = True
        self.initial_clustering = initial_clustering
        for t in range(self.T):
            inner_taxonomy_tracker = []
            inner_confusion_tracker = []
            s_l = []  # List of silhouette statistics
            h_l = []  # List of clustering according to k
            # To build a probabilistic penalty:
            probs_l = []  # List of clustering probabilities BEFORE DRAGGING
            range_to_loop = self.k_range  # To enable skipping other k's if we have a initialization
            if first_t and initial_clustering:
                range_to_loop = [max(initial_clustering) + 1]
            for k in range_to_loop:
                if first_t and initial_clustering:
                    clustered_cs = self.cs_clustering(t, k=k, CSmodel='initialization')
                else:
                    clustered_cs = self.cs_clustering(t, k=k, CSmodel='kmeans')
                X_bar = clustered_cs[0]
                h = clustered_cs[1]
                if X_bar.shape[1] < max(self.k_range):
                    unrealistic_array = np.array([self.unrealistic_centers for
                                                  _ in range(max(self.k_range) - X_bar.shape[1])]).T
                    X_bar = np.hstack((X_bar, unrealistic_array))
                if not first_t:
                    # Do the transition and relaxed assignment
                    # Rows are t-1, cols are t
                    confusion = confusion_matrix(list(self.H[:, t-1]), list(h),
                                                 labels=[i for i in range(max(self.k_range))])
                    # Transition (does not use a matrix but gives the indices)
                    P_row_ind, P_col_ind = linear_sum_assignment(-confusion)
                    X_bar = X_bar[:, P_col_ind]  # The Xbar tracks the changes in clusters
                    # Now relaxed assignment
                    # Does the cluster exist in this period?
                    h = [[j for j in P_col_ind if P_col_ind[j] == i][0] for i in h]  # Candidate
                    disappeared = [i not in list(h) for i in list(self.H[:, t - 1])]
                    disappeared = np.array([disappeared]).T
                    X_bar_i = (X_bar[:, list(self.H[:, t-1])].T * (1 - disappeared)) +\
                              (X_bar[:, list(h)].T * disappeared)

                    # X_bar_i has one row for every point, one column for every dimension:
                    relaxed_positions = self.CS_l[t] + epsilon*(X_bar_i - self.CS_l[t])
                    h = self.assignment(relaxed_positions, X_bar)

                    # Second-pass alignment
                    confusion = confusion_matrix(list(self.H[:, t - 1]), list(h),
                                                 labels=[i for i in range(max(self.k_range))])
                    P_row_ind, P_col_ind = linear_sum_assignment(-confusion)
                    X_bar = X_bar[:, P_col_ind]
                    h = [[j for j in P_col_ind if P_col_ind[j] == i][0] for i in h]
                    # Store mapping to create taxonomy after.
                    inner_taxonomy_tracker += [[P_row_ind, P_col_ind]]
                    inner_confusion_tracker += [confusion]
                else:
                    relaxed_positions = self.CS_l[t].copy()
                h_l += [h]

                # Now calculate the silhouette statistic to determine the optimal k
                s_l += [self.silhouette(self.CS_l[t], X_bar, h, actual_assignment=False)]
            # Now select best k:
            s_l_plain = s_l.copy()
            adj_s_l = [s * (1 - gini(np.array(list(Counter(h).values())).astype(float)))
                       for s, h in zip(s_l, h_l)]  # Using gini index
            adj_s_l_plain = [s * (1 - gini(np.array(list(Counter(h).values())).astype(float)))
                       for s, h in zip(s_l_plain, h_l)]  # Using gini index
            if gini_s:
                if len(adj_s_l) == 1:
                    s_star = 0
                else:
                    s_star = np.nanargmax(adj_s_l)
            else:
                if len(s_l) == 1:
                    s_star = 0
                else:
                    s_star = np.nanargmax(s_l)
            self.silhouette_stat += [s_l_plain[s_star]]
            self.gini_weighted_silhouette_stat += [adj_s_l_plain[s_star]]
            self.H[:, t] = h_l[s_star]
            # Calculate distance to cluster means (one cluster per column)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                cluster_centers = np.vstack([self.CS_l[t][[hh == i for hh in h_l[s_star]], :].mean(axis=0)
                                             for i in range(1 + max((h_l[s_star])))]).T
            centers_dist = cdist(self.CS_l[t], cluster_centers.T)
            self.distance_to_cluster_mean[:, t] = centers_dist[np.arange(len(centers_dist)), h_l[s_star]]
            if first_t:
                first_t = False  # Switch
            else:
                self.taxonomy_tracker += [inner_taxonomy_tracker[s_star]]
                self.confusion_tracker += [inner_confusion_tracker[s_star]]

            # Calculate the MSE
            X_bar_local = []
            for j in range(max(self.k_range)):
                if any(list(self.H[:, t] == j)):
                    X_bar_local += [np.mean(self.CS_l[t][self.H[:, t] == j, :], axis=0)]
                else:
                    X_bar_local += [np.zeros((1, self.CS_l[t].shape[1]))]
            X_bar_local = np.vstack(X_bar_local).T
            MSE += [np.mean(np.diag(cdist(self.CS_l[t], X_bar_local[:, list(self.H[:, t])].T)))]
        self.build_switches()
        self.CS_MSE = np.mean(MSE)
        return self.H

    def build_switches(self):
        """
        Builds the switches matrix out of the H matrix.
        """
        switches = np.zeros(self.H.shape).astype(int)
        for t in range(switches.shape[1] - 1):
            switches[:, t+1] = self.H[:, t] != self.H[:, t+1]
        self.switches = switches

    def taxonomy_builder(self, overlap=0.2):
        # Get number of clusters in each cross-section
        # rows are t-1
        cs_k = [list(set(self.H[:,i])) for i in range(self.H.shape[1])]
        cluster_counter = [Counter(self.H[:, i]) for i in range(self.H.shape[1])]
        # The row index of the confusion matrix is always the cluster index
        # 4-tupples of the cluster at t-1, the cluster at t, the weight of the link and the time t-1:
        connexions = []
        # for t in range(len(self.confusion_tracker)):
        for t in range(self.H.shape[1] - 1):
            cf = confusion_matrix(list(self.H[:, t]), list(self.H[:, t+1]),
                                  labels=[i for i in range(max(self.k_range))])
            cf = cf/cf.sum(1)[:, None]  # Will generate warnings
            cf[np.isnan(cf)] = 0
            for i in range(cf.shape[0]):
                if cf[i, :].sum() != 0:
                    flagged_connexions = [idx for idx, val in enumerate(cf[i, :] > overlap) if val]
                    connexions += [(i, j, cf[i, j], t) for j in flagged_connexions]
        return connexions, cs_k, cluster_counter

    def ward(self, maxclust=3):
        """ Performs Ward clustering (as a benchmark). """
        MSE = []
        first_t = True
        for t in range(self.T):
            linkage = hierarchy.ward(self.CS_l[t])
            # Careful with the -1 below;
            # It is there because the function returns cluster indices starting on 1, not 0
            h = hierarchy.fcluster(linkage, maxclust, 'maxclust') - 1
            self.H[:, t] = h
            if not first_t:
                # Do the transition
                # Rows are t-1, cols are t
                confusion = confusion_matrix(list(self.H[:, t-1]), list(self.H[:, t]),
                                             labels=[i for i in range(maxclust)])
                # Transition (does not use a matrix but gives the indices)
                P_row_ind, P_col_ind = linear_sum_assignment(-confusion.T)  # Note the .T
                self.taxonomy_tracker += [[P_row_ind, P_col_ind]]
                self.confusion_tracker += [confusion]
                self.H[:, t] = [P_col_ind[hh] for hh in h]
            else:
                first_t = False  # Switch
            # Calculate the MSE
            X_bar = np.vstack([np.mean(self.CS_l[t][h == j, :], axis=0) for j in range(maxclust)]).T
            MSE += [np.mean(np.diag(cdist(self.CS_l[t], X_bar[:, list(h)].T)))]
        self.build_switches()
        self.CS_MSE = np.mean(MSE)
        return self.H

    def recalculate_MSE(self, newdata, newident):
        """ To be used when doing naive Ward. """
        self.N = max([i[0] for i in newident]) + 1
        self.T = max([i[1] for i in newident]) + 1
        self.ident = newident
        self.data = newdata
        self.CS_l = [i for i in self._generate_cs(newdata)]
        MSE = []
        h = self.H[:, 0]
        maxclust = max(h + 1)
        for t in range(self.T):
            X_bar = np.vstack([np.mean(self.CS_l[t][h == j, :], axis=0) for j in range(maxclust)]).T
            MSE += [np.mean(np.diag(cdist(self.CS_l[t], X_bar[:, list(h)].T)))]
        self.CS_MSE = np.mean(MSE)


def kmeans(data, k=2, chg_threshold=0.02):
    """
    k-means clustering for a predefined number k of clusters. Simplified version based on ward_fcts.py.

    Args:
        data (numpy.ndarray): Array of one point in each row.
        k (int): Number of clusters.
        chg_threshold (float): Threshold in relative change (wrt the initial distance between centers) to stop
            iterating.

    Returns:
        numpy.ndarray: Array with one of k centers in each column.
    """

    # Create distance matrix (euclidean)
    m_dist = squareform(pdist(data))  # condensed matrix or squareform?

    # Choose initial cluster centers
    # Here there are many possibilities. I choose the 2 points farthest apart.
    centers_ind = list(np.unravel_index(np.argmax(m_dist), m_dist.shape))
    centers = data[centers_ind]

    # After the first two, I select points that have the largest sum of distances to the other centers.
    if k > 2:
        remaining_data = data
        for i in range(2, k):
            remaining_data = remaining_data[[j for j in range(remaining_data.shape[0]) if j not in centers_ind]]
            centers_ind = [np.argmax(np.sum(cdist(remaining_data, centers), axis=1))]
            centers = np.vstack((centers, remaining_data[centers_ind, :]))

    # Initialize other variables
    rel_change = chg_threshold + 1
    cluster_assignments = []
    # iter_count = 0
    # To calculate relative change we need the minimal distance between centers at the beginning:
    initial_dist = np.min(pdist(centers))

    while rel_change > chg_threshold:
        # Now we need to calculate what is the closest center to each point in the data, and allocate them.
        centers_dist = cdist(data, centers)
        cluster_assignments = np.array([np.argmin(centers_dist[i, :]) for i in range(centers_dist.shape[0])])
        # Recalculate centers
        new_centers = np.array([np.mean(data[cluster_assignments == i, :], axis=0) for i in range(k)])
        # Check change rate
        # Distance cover by this iteration
        change = np.diag(cdist(new_centers, centers))
        # Relative change
        rel_change = np.max(change / initial_dist)
        # Finalize loop and store history
        centers = new_centers
        # iter_count += 1

    else:
        return centers.T, cluster_assignments


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 1e-6
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def group_means(X, h, fct='mean'):
    # One row for each group, one col for each variable.
    # Assumes indexing of clusters in h starts at 0
    if fct == 'mean':
        means_m = np.full((h.max() + 1, X.shape[1]), np.Inf)
        for i in list(np.unique(h)):
            means_m[i, :] = np.mean(X[h == i, :])
        to_ret = means_m
    if fct == 'var':
        covar_d = {i: np.diag(np.var(X[h == i, :], axis=0)) for i in list(np.unique(h))}
        for i in list(np.unique(h)):
            if sum(h == i) == 1:
                covar_d[i] = np.cov(X, rowvar=False)
        to_ret = covar_d
    return to_ret


def gabriel_CV(data, ident, n_folds_units=3, n_folds_vars=2, clustering_fct=None,
               classification_style='mean', par=0.5, k_range=[2, 3, 4]):
    # Gabriel cross-validation for unsupervised learning of Fu and Perry 2017
    # Returns the CV error
    # 4 steps: cluster, classify, predict, and evaluate
    if clustering_fct is None:
        def clustering_fct(dta, ide):
            modelo = CLModel(dta, ide, k_range=k_range)
            modelo.elastic(epsilon=par)
            return modelo.H
    n_units = max([i[0] for i in ident]) + 1
    n_periods = max([i[1] for i in ident]) + 1
    n_vars = data.shape[1]
    units_list = [i[0] for i in ident if i[1] == ident[0][1]]
    vars_list = list(range(data.shape[1]))
    # Fold assignment
    units_fold_assignment = [np.mod(i, n_folds_units) for i in range(n_units)]
    np.random.shuffle(units_fold_assignment)
    vars_fold_assignment = [np.mod(i, n_folds_vars) for i in range(n_vars)]
    np.random.shuffle(vars_fold_assignment)
    folds_list = list(itertools.product(np.arange(n_folds_units), np.arange(n_folds_vars)))
    CV_error = []
    for fold in folds_list:
        units_in_this_fold = [i for i, j in zip(units_list, units_fold_assignment) if j == fold[0]]
        rows_in_this_fold = [ind for ind, ide in enumerate(ident) if (ide[0] in units_in_this_fold)]
        units_not_in_this_fold = [i for i, j in zip(units_list, units_fold_assignment) if j != fold[0]]
        rows_not_in_this_fold = [ind for ind, ide in enumerate(ident) if (ide[0] in units_not_in_this_fold)]
        # Train data
        data_train = data[rows_in_this_fold]
        ident_dict_new_to_old_train = dict([(ind, i) for ind, i in enumerate(units_in_this_fold)])
        ident_dict_old_to_new_train = {v: k for k, v in ident_dict_new_to_old_train.items()}
        ident_train = [ide for ide in ident if (ide[0] in units_in_this_fold)]
        ident_train = [(ident_dict_old_to_new_train.get(i[0]), i[1]) for i in ident_train]
        Y_vars_in_this_fold = [i for i, j in zip(vars_list, vars_fold_assignment) if j == fold[1]]
        X_vars_in_this_fold = [i for i, j in zip(vars_list, vars_fold_assignment) if j != fold[1]]
        Y_train = data_train[:, Y_vars_in_this_fold]
        X_train = data_train[:, X_vars_in_this_fold]
        # Test data
        data_test = data[rows_not_in_this_fold]
        ident_dict_new_to_old_test = dict([(ind, i) for ind, i in enumerate(units_not_in_this_fold)])
        ident_dict_old_to_new_test = {v: k for k, v in ident_dict_new_to_old_test.items()}
        Y_test = data_test[:, Y_vars_in_this_fold]
        X_test = data_test[:, X_vars_in_this_fold]
        ident_test = [ide for ide in ident if (ide[0] in units_not_in_this_fold)]
        ident_test = [(ident_dict_old_to_new_test.get(i[0]), i[1]) for i in ident_test]
        # Run clustering and retrieve assignments and means
        h = clustering_fct(Y_train, ident_train)
        # Now classify step using the assignments and the the X
        if classification_style == 'mean':
            # First retrieve time-varying means
            center = {t: group_means(X_train[[ide[1] == t for ide in ident_train]], h[:, t]) for
                      t in range(n_periods)}
            center_Y = {t: group_means(Y_train[[ide[1] == t for ide in ident_train]], h[:, t]) for
                      t in range(n_periods)}
            # Run prediction on the testing dataset
            h_test = np.full((len(units_not_in_this_fold), n_periods), 0)
            Y_test_hat = []
            for t in range(n_periods):
                X_test_t = X_test[[ide[1] == t for ide in ident_test]]
                h_test[:, t] = np.argmin(cdist(X_test_t, center.get(t)), axis=1)  # Euclidian distance
                # Now predict the Y test hat
                Y_test_hat += [center_Y.get(t)[h_test[:, t], :]]
        if classification_style == 'epsilon':
            # First retrieve time-varying means
            center = {t: group_means(X_train[[ide[1] == t for ide in ident_train]], h[:, t]) for
                      t in range(n_periods)}
            center_Y = {t: group_means(Y_train[[ide[1] == t for ide in ident_train]], h[:, t]) for
                      t in range(n_periods)}
            # Run prediction on the testing dataset
            h_test = np.full((len(units_not_in_this_fold), n_periods), 0)
            Y_test_hat = []
            for t in range(n_periods):
                X_test_t = X_test[[ide[1] == t for ide in ident_test]]
                distances = cdist(X_test_t, center.get(t))
                if t > 0:
                    drag = np.ones(distances.shape)
                    for i, hh in enumerate(list(h_test[:, t - 1])):
                        if hh < drag.shape[1]:
                            # If the last cluster has disappeared, dont drag anywhere
                            drag[i, hh] = par
                    # Will generate warning when it contains inf and nan, but is expected.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        distances = np.multiply(distances, drag)
                distances[np.isnan(distances)] = np.Inf
                h_test[:, t] = np.argmin(distances, axis=1)  # Euclidian distance
                # Now predict the Y test hat
                Y_test_hat += [center_Y.get(t)[h_test[:, t], :]]
        Y_test_hat = np.vstack(Y_test_hat)
        # Evaluate step
        CV_error += [np.mean(paired_distances(Y_test_hat, Y_test)**2)]
    return np.mean(CV_error)
