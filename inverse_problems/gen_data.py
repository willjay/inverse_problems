"""
Generates a spectral function to simulate and saves it to an hdf5 file.

Arguments
---------
string : path to save the spectral function at.
"""

import sys
import numpy as np
import gmpy2 as gmp
from fileio import *
# from nevanlinna import *

sys.path.append('/Users/theoares/lqcd/spectral/python_scripts')
from utils import *

# Set precision for gmpy2 and initialize complex numbers
prec = 128
gmp.get_context().allow_complex = True
gmp.get_context().precision = prec
I = gmp.mpc(0, 1)

fname = str(sys.argv[1])
print('Writing data to: ' + fname)

def matsubara(beta, boson = False):
    rng = np.arange(beta)
    if boson:
        return np.array([2*gmp.const_pi()*I*n/beta for n in rng])
    return np.array([(2*n + 1)*gmp.const_pi()*I/beta for n in rng])

def pole(m, z):
    return 1 / (m - z)

def analytic_dft(m, z, beta):
    prefactor = -1*(1 + np.exp(-m*beta))
    pole1 = 1/(np.exp(z-m) - 1)
    pole2 = 1/(np.exp(z+m) - 1)
    return prefactor * (pole1 + pole2)

def analytic_ft(z, m, beta):
    prefactor = -1*(1 + gmp.exp(-m*beta))
    pole1 = 1/(z - m)
    pole2 = 1/(z + m)
    return prefactor * (pole1 + pole2)

def unstable_pole(z, m, gamma):
    mpi = 0.140
    return 0.25*(1 - 4*mpi**2/z**2)**0.5/(z - (m - 1j*gamma))

def kinematic_feature(z):
    z = z + 0*1j
    return -1*(0.25 - z)**0.5

# Construct desired spectral function
beta = 48
freqs = matsubara(beta, boson = True)
ngs = np.zeros(len(freqs), dtype = object)
for mm in [gmp.mpfr("0.05"), gmp.mpfr("0.1")]:
    ngs = ngs + analytic_ft(freqs, mm, beta)

freqs, ngs = freqs[1:], ngs[1:]

print('Freqs: ')
print(freqs)

print('Greens function data: ')
print(ngs)

phis = construct_phis(freqs, h(ngs))

print('Phis: ')
print(phis)

# Save spectral function
write_gmp_input_h5(fname, beta, freqs, ngs)
print('Green\'s function data written to: ' + fname)
