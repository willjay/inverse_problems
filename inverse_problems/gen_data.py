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

sys.path.append('/Users/theoares/lqcd/spectral/python_scripts')
from utils import *

sys.path.append('/Users/theoares/lqcd/utilities')
from formattools import *
import plottools as pt
sty = styles['talk']
pt.set_font()

# Set precision for gmpy2 and initialize complex numbers
# prec = 128
prec = 256
gmp.get_context().allow_complex = True
gmp.get_context().precision = prec
I = gmp.mpc(0, 1)

fname = str(sys.argv[1])
plot_dir = str(sys.argv[2])
print('Writing data to: ' + fname)
print('Plotting data at path: ' + plot_dir)

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
beta = 64
# beta = 256
freqs = matsubara(beta, boson = True)
ngs = np.zeros(len(freqs), dtype = object)
# for mm in [gmp.mpfr("0.05"), gmp.mpfr("0.1")]:
#     ngs = ngs + analytic_ft(freqs, mm, beta)
for mm in [gmp.mpfr('0.04'), gmp.mpfr('0.08'), gmp.mpfr('0.12')]:
    ngs = ngs + analytic_ft(freqs, mm, beta)
# for mm in [gmp.mpfr('0.04'), gmp.mpfr('0.06'), gmp.mpfr('0.1'), gmp.mpfr('0.14'), gmp.mpfr('0.16')]:
#     ngs = ngs + analytic_ft(freqs, mm, beta)
# for mm in [gmp.mpfr('0.04'), gmp.mpfr('0.06'), gmp.mpfr('0.08'), gmp.mpfr('0.1'), gmp.mpfr('0.14'), gmp.mpfr('0.16')]:
#     ngs = ngs + analytic_ft(freqs, mm, beta)
# for mm in [gmp.mpfr(x) for x in ['0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.10', '0.11', '0.12', '0.13', '0.14', '0.15', '0.16', '0.17', '0.18']]:
#     ngs = ngs + analytic_ft(freqs, mm, beta)

freqs, ngs = freqs[1:], ngs[1:]

# Subset frequencies if desired
sub_idxs = range(0, len(freqs), 2)
print('Subsetting at indices: ' + str(sub_idxs))
freqs = freqs[sub_idxs]
ngs = ngs[sub_idxs]

print('Generating data with ' + str(len(sub_idxs)) + ' Matsubara frequencies.')

print('Freqs: ')
print(freqs)

print('Greens function data: ')
print(ngs)

# phis = construct_phis(freqs, h(ngs))
# print('Phis: ')
# print(phis)

# Save spectral function
write_gmp_input_h5(fname, beta, freqs, ngs)
print('Green\'s function data written to: ' + fname)

# plot Green's function.
gfn_plot_path = plot_dir + 'greens_fn_data.pdf'
pt.plot_1d_points(np.imag(np.complex64(freqs)), np.imag(np.complex64(ngs)), col='b', ax_label=[r'$i\omega$', r'$G(i\omega)$'], title='Green\'s function.',
                  style = sty, saveat_path = gfn_plot_path)#, logy = True)
print('Plot of Green\'s function data saved to: ' + gfn_plot_path)

# plot spectral function. TODO