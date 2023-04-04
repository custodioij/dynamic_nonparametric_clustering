"""
Code used in:
    Igor Custodio João, Julia Schaumburg, André Lucas, Bernd Schwaab, Dynamic Nonparametric Clustering of Multivariate Panel Data, Journal of Financial Econometrics, 2022.
    https://doi.org/10.1093/jjfinec/nbac038
    
Function to simulate the data.
"""
import numpy as np
from scipy import stats as sts


def random_switches_multivar(N, T, n_vars=6, n_clusters=2, p=0.1, scaling=1):
    """
    Generates multivariate data with random switching to replicate the simulation section.

    Args:
        N (int): Number of units.
        T (int): Number of time steps.
        n_vars (int): Number of variables.
        n_clusters (int): Number of true clusters in the data.
        p (float): Switching rate.
        scaling (float): Scale of the covariance matrix.

    Returns:
        tupple: Returns a tupple (data, ident, assignments, switches) where:
            data: A Numpy array of the data matrix, with one column for each variable.
            ident: A list with one element for each row of data. Each element contains a list [i, t] where i is the unit
            marker and runs from 0 to N-1, and t is the time marker and runs from 0 to T-1.
            assignments: A Numpy array of the assignments matrix, with N rows and T columns, and where element [i, t] is
            the cluster assignment of unit i at time t, ranging from 0 to n_clusters-1.
            switches: A Numpy array of the switches matrix, with N rows and T columns, and where element [i, t] is
            equal to 1 if unit i changed clusters at time t. This can be one at t=0 as assigments are generated
            internally at t=-1.

    """

    # Generate cluster means:
    mus = ((np.arange(2 ** n_vars)[:, None] &
            (1 << np.arange(n_vars))) > 0).astype(float)
    idx = np.random.randint(mus.shape[0], size=n_clusters)
    mus = mus[idx, :]

    # cov = (np.full((n_vars, n_vars), 0.5) + (np.identity(n_vars) * 0.5)) * scaling
    cov = (np.identity(n_vars)) * scaling

    assignments = np.vstack([[np.mod(i, n_clusters) for i in range(N)] for _ in range(T)]).T
    switches = np.array(sts.bernoulli.rvs(p=p, size=N*T)).reshape(assignments.shape)
    # Change assignments:
    for t in range(T):
        # Switch in circles
        assignments[switches[:, t] == 1, t:] = (assignments[switches[:, t] == 1, t:] + 1) % n_clusters
    stacked_assignments = assignments.T.flatten().reshape((N*T, 1))

    clusters = [sts.multivariate_normal.rvs(mean=mu, cov=cov, size=N*T) for mu in mus]

    data = np.vstack([clusters[stacked_assignments[i][0]][i, :] for i in range(N*T)])
    ident = []
    for t in range(T):
        ident += [(i, t) for i in range(N)]

    return data, ident, assignments.astype(int), switches
