# Dynamic Nonparametric Clustering of Multivariate Panel Data
Code to reproduce the results of Dynamic Nonparametric Clustering of Multivariate Panel Data (Custodio João, Schaumburg, Lucas and Schwaab, 2022, Journal of Financial Econometrics, <https://doi.org/10.1093/jjfinec/nbac038>).

Running `Simulate_clustering.py` will generate a csv files with simulations results. This is used by `Simulation_plots.py` to reproduce Figure 5 of the paper.

`elastico.py` constains all functions needed for the clustering algorithm.
`simulate_fcts.py` contain the function that simulates the data.

`elastico.py` can be used to cluster other datasets. Typical usage might be:
```
model = el.CLModel(data, ident)
H = model.elastic(epsilon=epsilon)
```
where `epsilon` is the stickyness parameter, `data` is a numpy 2-way array with columns for variables and rows for observations, and the `ident` list of lists contains the indices $(i, t)$ of each row. For example, if $N=2$, $T=3$, and the number of dimensions $D=4$, these could be:
```
data = np.array([[0.7, 0.9, 0.1, 0.9],
                 [0.1, 1. , 0.1, 0.6],
                 [0. , 0.5, 0.9, 0.5],
                 [0.5, 0.3, 0.2, 0.5],
                 [0. , 0. , 0.4, 0.5],
                 [0.7, 0.6, 0.4, 0.7]])
ident = [[0,0], [0,1], [0,2],
         [1,0], [1,1], [1,2]]
```
