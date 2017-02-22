"""
Script for plotting solver performance.
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle
import shelve
import palettable


# Loop over formats
for format in ['csr', 'csc']:

    # Open data
    with shelve.open('timings_%s.dat' %format, 'r') as file:
        loops = file['loops']
        format = file['format']
        resolutions = file['resolutions']
        sizes = file['sizes']
        solver_names = file['solver_names']
        solver_docs = file['solver_docs']
        start_times = file['start_times']
        solve_times = file['solve_times']

    # Plot overlayed
    colors = palettable.colorbrewer.qualitative.Dark2_8.mpl_colors
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

    ax.set_color_cycle(colors)
    for solver_name, solver_doc, start_time, solve_time in zip(solver_names, solver_docs, start_times, solve_times):
        ax.loglog(sizes, solve_time, '.-', label=solver_name)
    ax.legend(loc='lower right', prop={'size':8})
    ax.set_title('Solve time')
    ax.set_xlabel('Matrix size')
    ax.set_ylabel('Time per solve (averaged over %i) (s)' %loops)
    ax.set_xlim([1e1, 1e6])
    ax.set_ylim([1e-5, 1e-0])
    ax.grid()
    fig.savefig('timing_solve_%s.pdf' %(format), dpi=100)

    colors = palettable.colorbrewer.qualitative.Dark2_8.mpl_colors
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

    ax.set_color_cycle(colors)
    for solver_name, solver_doc, start_time, solve_time in zip(solver_names, solver_docs, start_times, solve_times):
        ax.loglog(sizes, start_time, '.-', label=solver_name)
    ax.legend(loc='lower right', prop={'size':8})
    ax.set_title('Start time')
    ax.set_xlabel('Matrix size')
    ax.set_ylabel('Time per solve (averaged over %i) (s)' %loops)
    ax.set_xlim([1e1, 1e6])
    ax.set_ylim([1e-5, 1e-0])
    ax.grid()
    fig.savefig('timing_start_%s.pdf' %(format), dpi=100)
