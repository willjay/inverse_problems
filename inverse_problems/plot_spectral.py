"""
Plots a spectral function from an HDF5 file.

Arguments
---------
string : HDF5 file with spectral function reconstruction data.
string : directory to plot output in. 
"""
import sys
import numpy as np
import gmpy2 as gmp
from fileio import *

sys.path.append('/Users/theoares/lqcd/utilities')
from formattools import *
import plottools as pt
sty = styles['talk']
pt.set_font()

prec = 128
gmp.get_context().allow_complex = True
gmp.get_context().precision = prec
I = gmp.mpc(0, 1)

fname = str(sys.argv[1])
out_dir = str(sys.argv[2])
print('Using data at: ' + fname)
print('Plotting data in directory: ' + out_dir)

beta, start, stop, num, eta, freqs, ng, recon, abcd = read_gmp_output_h5(fname)

print('Output params')
print(beta, start, stop, num)

# print('abcd matrix')
# print(abcd)

plt_path = out_dir + 'recon.pdf'
xx = np.linspace(start, stop, num)
recon_double = np.complex64(recon)
pt.plot_1d_points(xx, np.imag(recon_double), col='b', ax_label=[r'$\omega$', r'$\rho(\omega)$'], \
    title = r'Reconstructed Spectral Function ($\eta = ' + str(np.float64(eta)) + r'$)', style = sty, saveat_path = plt_path, logy = True)
print('Plot of reconstruction saved to: ' + plt_path)
