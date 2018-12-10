"""Plot solver performance."""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import xarray
import pathlib


# Load timing files
timing_files = list(pathlib.Path('.').glob('timing_*.pkl'))
ds = []
for timing_file in timing_files:
    dss = pickle.load(open(str(timing_file), 'rb'))
    dss = dss.expand_dims('solver_name')
    dss = dss.assign_coords(solver_name=[dss.attrs['solver_name']])
    ds.append(dss)
ds = xarray.concat(ds, dim='solver_name')
ds = ds.sortby('solver_name')

plt.clf()
xarray.plot.plot(ds['start_time'].isel(bw=0),
                 hue='solver_name', col='format',
                 xscale='log', yscale='log', marker='.')
plt.savefig('start_time_bw_min.pdf')

plt.clf()
xarray.plot.plot(ds['start_time'].isel(Nz=-1),
                 hue='solver_name', col='format',
                 xscale='log', yscale='log', marker='.')
plt.savefig('start_time_Nz_max.pdf')

plt.clf()
xarray.plot.plot(ds['solve_time'].isel(bw=0),
                 hue='solver_name', col='format',
                 xscale='log', yscale='log', marker='.')
plt.savefig('solve_time_bw_min.pdf')

plt.clf()
xarray.plot.plot(ds['solve_time'].isel(Nz=-1),
                 hue='solver_name', col='format',
                 xscale='log', yscale='log', marker='.')
plt.savefig('solve_time_Nz_max.pdf')

