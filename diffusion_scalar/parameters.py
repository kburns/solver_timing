"""
Parameters for solver test.
"""

import numpy as np
from collections import OrderedDict

# Matrix parameters
args = OrderedDict()
args['Nz'] = 2**np.arange(6, 10)
args['bw'] = 2**np.arange(6)
args['format'] = ['csr', 'csc']

# Timing parameters
loops = 10

