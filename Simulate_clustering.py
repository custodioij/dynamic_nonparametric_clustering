"""
Code to reproduce Figure 5 from:
    Igor Custodio João, Julia Schaumburg, André Lucas, Bernd Schwaab, Dynamic Nonparametric Clustering of Multivariate Panel Data, Journal of Financial Econometrics, 2022.
    https://doi.org/10.1093/jjfinec/nbac038

After running this script, run Simulation_plots.py to produce the plots.

This script will save as many csv files as the length of the list simulation_settings_l, which are used by Simulation_plots.py.

Enabling the cross-validation (top-left panel of Figure 5 in the paper) increases the running time considerably. Uncomment the relevant lines to enable.
"""


import elastico as el
import simulate_fcts as simulate
import pandas as pd
import numpy as np
import time
from itertools import chain
from sklearn.metrics import confusion_matrix  # Build the confusion matrix
from scipy.optimize import linear_sum_assignment  # For Hungarian Algorithm

# Settings to loop around:
# sigma_scale is the variance and k_range is a list of number of clusters to try to estimate.
# Uncomment the other settings for other plots from the paper.
simulation_settings_l=[
                    #  {'simulation_name': 'two_cl_var_k_large_N_k_6_S50',
                    #   'sigma_scale': 0.5,
                    #   'k_range': [2, 3, 4]},
                    #  {'simulation_name': 'two_cl_var_k_large_N_k_6',
                    #   'sigma_scale': 1,
                    #   'k_range': [2, 3, 4]},
                     {'simulation_name': 'two_cl_fix_k_large_N_k_6_S50',
                      'sigma_scale': 0.5,
                      'k_range': [2]} #,
                    #  {'simulation_name': 'two_cl_fix_k_large_N_k_6',
                    #   'sigma_scale': 1,
                    #   'k_range': [2]}
                     ]
n_clusters = 2  # Number of true clusters to be generated.
n_vars = 6  # Number of variables in the data.
N = int(120)  # Number of units
T = int(20)  # Number of time steps.
p_l = [0, 0.01, 0.1, 0.25]  # List with the probabilitie of switching to loop around.
B = int(100)  # Number of simulation runs per setting per value of p.
epsilon_l = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # Values of the epsilon parameter to be tried.

def missclassification(H, assign):
    # Use the same strategy as in the main clustering with the Hungarian algorithm.
    labels_in_H = list(chain(*H.tolist()))
    labels_in_assign = list(chain(*assign.tolist()))
    cost_matrix = -confusion_matrix(labels_in_H, labels_in_assign)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    min_miss = 1 + cost_matrix[row_ind, col_ind].sum()/len(labels_in_assign)
    return min_miss


def off_diagonal_miss(switches_model, switches_true):
    cf = confusion_matrix(switches_true.flatten(), switches_model.flatten(), labels=[0, 1])
    false_positives = cf[0, 1]
    true_positives = cf[1, 1]
    false_negatives = cf[1, 0]
    true_negative = cf[0, 0]
    return false_positives, true_positives, false_negatives, true_negative


def simulation_run(p, simulation_settings):
    sigma_scale = simulation_settings['sigma_scale']
    k_range = simulation_settings['k_range']
    model = 'eps'
    results_frame = pd.DataFrame(columns=['par', 'model', 'p', 'b', 'miss',
                                          'false_pos', 'true_pos', 'false_neg', 'true_neg',
                                          'CV_error_par', 'Gini_weighted_silhouette'])
    results_el = results_frame.copy()
    for b in range(B):
        results_el_b = results_frame.copy()
        print('b = ' + str(b) + '; p = ' + str(p))

        data, ident, assignments, switches = \
            simulate.random_switches_multivar(N=N, T=T, n_vars=n_vars, n_clusters=n_clusters,
                                                    p=p, scaling=sigma_scale)

        count = 0
        for par in epsilon_l:
            modelo = el.CLModel(data, ident, k_range=k_range)
            def clustering_fct(dta, ide):
                modelo = el.CLModel(dta, ide, k_range=k_range)
                modelo.elastic(epsilon=par)
                return modelo.H
            modelo.elastic(epsilon=par)
            switch_eps = modelo.switches
            fp, tp, fn, tn = off_diagonal_miss(switch_eps, switches)
            miss = missclassification(modelo.H, assignments)
            adj_s_stat = np.mean(modelo.gini_weighted_silhouette_stat)
            CV_error_par = None
            # Uncomment below to enable cross-validation
            # try:
            #     CV_error_par = el.gabriel_CV(data, ident, clustering_fct=clustering_fct,
            #                                  classification_style='epsilon', par=par, k_range=k_range)
            # except:
            #     print('Problem in the CV par')
            results_el_b.loc[count] = [par, model, p, b, miss,
                                       fp, tp, fn, tn,
                                       CV_error_par, adj_s_stat]
            count += 1
            del modelo

        results_el = pd.concat([results_el, results_el_b])

    return results_el


if __name__ == '__main__':
    t0 = time.time()
    for simulation_settings in simulation_settings_l:
        simulation_name = simulation_settings['simulation_name']
        def simulation_run_temp(p):
            return simulation_run(p, simulation_settings)
        results_el_list = []
        for p, res in zip(p_l, map(simulation_run_temp, p_l)):
            results_el_list += [res]
        results_el = pd.concat(results_el_list)
        results_el.to_csv('MC_random_switches_B_' + str(B) +
                          '_' + simulation_name + '.csv')

        print('saved as ' + 'MC_random_switches_B_' + str(B) +
              '_' + simulation_name + '.csv')
    t1 = time.time()
    print("Running time: " + str((t1 - t0) / 60) + ' minutes.')


