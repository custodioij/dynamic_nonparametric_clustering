"""
Code to reproduce Figure 5 from:
    Igor Custodio João, Julia Schaumburg, André Lucas, Bernd Schwaab, Dynamic Nonparametric Clustering of Multivariate Panel Data, Journal of Financial Econometrics, 2022.
    https://doi.org/10.1093/jjfinec/nbac038

Run this script after Simulate_clustering.py.
This script will save as many figures as the lenght of list simulation_name_l.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Names of the simulation results files:
# Uncomment the other settings if they were run in Simulate_clustering.py.
simulation_name_l = ['B_100_two_cl_fix_k_large_N_k_6_S50']#,
                     # 'B_100_two_cl_var_k_large_N_k_6_S50',
                     # 'B_100_two_cl_fix_k_large_N_k_6',
                     # 'B_100_two_cl_var_k_large_N_k_6']

for simulation_name in simulation_name_l:
    df = pd.read_csv('MC_random_switches_' + simulation_name + '.csv')
    df = df.drop(df.columns[0], axis=1)
    df['switches'] = (df['true_pos'] + df['false_pos'])
    df['switches'] = df['switches'] / df['switches'].max()
    df['FPR'] = df['false_pos'] / (df['false_pos'] + df['true_neg'])
    df['TPR'] = df['true_pos'] / (df['true_pos'] + df['false_neg'])
    df_summarized = df.fillna(0).groupby(['p', 'model', 'par']).mean(numeric_only=True).drop('b', axis=1)
    df_summarized.to_csv('MC_random_switches_flickering_' + simulation_name + '_summary.csv')
    df_summarized = df_summarized.reset_index()
    df_summarized['ROC_diff'] = df_summarized['TPR'] - df_summarized['FPR']

    """ Figures for the paper """
    results = df_summarized[df_summarized['model'] == 'eps']
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    variables_to_plot = ['miss', 'CV_error_par', 'switches', 'Gini_weighted_silhouette']
    plot_title = ['Misclassification rate', 'CV error', 'Switching rate', 'Gini weighted silhouette']
    for name, grp in results.groupby(['p']):
        for i in range(len(variables_to_plot)):
            variable = variables_to_plot[i]
            grp.plot(ax=axs[i], kind='line', x='par', y=variable, label=str(name))
            axs[i].set_title(plot_title[i])
            axs[i].set_xlabel(r'$\epsilon$')
            axs[i].legend(title=r'$p$')
            if i>0: axs[i].get_legend().remove()
    model_name_title = 'Epsilon'
    plt.savefig("eps_to_print_" + simulation_name + ".png", bbox_inches='tight')
