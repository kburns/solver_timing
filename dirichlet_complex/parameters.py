"""
Parameters for solver test.
"""

import numpy as np
from collections import OrderedDict

# Matrix parameters
args = OrderedDict()
args['Nz'] = 2**np.arange(6, 9)
args['bw'] = np.arange(5)
args['format'] = ['csr', 'csc']

# Timing parameters
loops = 50
sparse_only = False

