"""
Implements Nevalinna analytic continuation, mostly following the ideas from
Jiani Fei, Chia-Nan Yeh, Emanuel Gull
"Nevanlinna Analytical Continuation"
Phys. Rev. Lett. 126, 056402 (2021)
arXiv: [https://arxiv.org/abs/2010.04572]
"""
import numpy as np


def moebius(z):
    return (z - 1j) / (z + 1j)


def inverse_moebius(z):
    return 1j * (1 + z) / (1 - z)


def theta_map(mat, z):
    """
    Evaluates the conformal map "theta" defined by a matrix [a,b],[c,d]
    theta = (a*z + b)/(c*z + d).
    """
    [a, b], [c, d] = mat
    return (a*z + b)/(c*z + d)


def pick_matrix(freq, ng):
    if len(freq) != len(ng):
        raise ValueError("Incommensurate sizes", len(freq), len(ng))
    npts = len(freq)
    z_upper = moebius(ng)
    z_lower = moebius(freq)
    pick = np.zeros((npts, npts), dtype=complex)
    for i in range(npts):
        for j in range(npts):
            pick[i, j] =\
                (1 - z_upper[i] * np.conjugate(z_upper[j]))/\
                (1 - z_lower[i] * np.conjugate(z_lower[j]))
    return pick


class ImaginaryDomainData:
    def __init__(self, freq, ng):
        """
        Container for imaginary-domain data
        Args:
            freq: the values of the Matsubara frequencies
            ng: the values of the Nevanlinna Green function
        """
        freq = np.array(freq)
        ng = np.array(ng)
        if len(freq) != len(ng):
            raise ValueError("Incommensurate data", len(freq), len(ng))
        if not np.allclose(np.abs(freq.real), 0):
            raise ValueError("Matsubara frequencies mus be purely imaginary")
        self.freq = freq
        self.ng = ng
        self.h = moebius(ng)

    def __len__(self):
        return len(self.freq)


class RealDomainData:
    def __init__(self, start=0, stop=1, num=50, eta=1e-3):
        """
        Container for real-domain data.
        Args:
            start, stop, num: args for linspace
            eta: float, the small imaginary "regulator"
        """
        self.freq = np.linspace(start, stop, num) + 1j*eta


class Schur:
    """
    Interpolation of Schur functions (holomorphic functions from the open
    unit disk to the closed unit disk) using continued fractions expressed in
    terms of Moebius transformations.
    """
    def __init__(self, imag):
        """
        Args:
            imag: ImaginaryDomainData
        """
        self.imag = imag
        self.phi = np.zeros(len(imag), dtype=complex)
        self.npts = len(self.imag)
        self.initialize()

    def gamma(self, n, z):
        """
        Computes gamma_n = (z - y_n)/(z - y_n^{*})
        """
        y = self.imag.freq
        return (z - y[n]) / (z - np.conjugate(y[n]))

    def theta_matrix(self, idx, z):
        """
        Computes the matrix associated with the nth Moebius tranformation,
        theta_n = [[gamma_n(z), phi_n], [gamma_n(z) phi_n^{*}, 1]
        """
        gamma = self.gamma(idx, z)
        result = np.array(
            [[gamma, self.phi[idx]],
             [np.conjugate(self.phi[idx]) * gamma, 1.0]],
            dtype=complex)
        # result = np.zeros((2, 2), dtype=complex)
        # result[0, 0] = gamma
        # result[0, 1] = self.phi[idx]
        # result[1, 0] = np.conjugate(self.phi[idx]) * gamma
        # result[1, 1] = 1.0
        return result

    def initialize(self):
        """
        Computes the interpolation parameters phi[j] inductively.
        """
        y = self.imag.freq
        self.phi[0] = self.imag.h[0]
        for j in range(1, self.npts):
            # Compute the product of matrices in Eq.(8), evaluated at z=y[j]
            arr = np.eye(2, dtype=complex)
            for k in range(0, j):
                arr = arr @ self.theta_matrix(idx=k, z=y[j])
            # Solve for phi[j] in terms of "theta[j-1]"
            [a, b], [c, d] = arr
            self.phi[j] = (self.imag.h[j]*d - b) / (a - self.imag.h[j]*c)


    def thetas(self, z):
        """
        Computes the list of matrices [theta_n(z)] which define the Moebius
        maps used in the interpolation.
        """
        return [self.theta_matrix(idx=idx, z=z) for idx in range(self.npts)]

    # def __call__(self, z, map_back=True):
    def __call__(self, z, fcn=None, map_back=True, **kwargs):
        """
        Evaluate the interpolant.
        Args:
            z: complex, the value at which to compute the interpolant
            map_back: bool, whether to map the result from the unit disk back
                to the upper half plane.
        """
        # Choose theta_M to be a vanishing constant
        # TODO: optimize with Hardy functions or some other method
        # param = 0.
        if fcn is None:
            def _fcn(z, **kwargs):
                return 0
            fcn = _fcn
        # Compute the product of matrices in Eq.(8)
        (a,b), (c,d) = (1, 0), (0, 1)
        arr = np.eye(2, dtype=complex)
        for idx in range(self.npts):
            arr = arr @ self.theta_matrix(idx=idx, z=z)
        # result = theta_map(arr, param)
        # for idx in range(self.npts):
        #     gamma = self.gamma(idx, z)
        #     phi = self.phi[idx]
        #     A = gamma
        #     B = phi
        #     C = np.conjugate(phi) * gamma
        #     D = 1.0
        #     anew = a*A + b*C
        #     bnew = a*B + b*D
        #     cnew = c*A + d*C
        #     dnew = c*B + d*D
        #     a = anew
        #     b = bnew
        #     c = cnew
        #     d = dnew
        # arr = np.array([[a, b], [c, d]])
        arr = arr / np.max(arr)  # Normalize entries
        result = theta_map(arr, fcn(z, **kwargs))
        if not map_back:
            return result
        return inverse_moebius(result)


class Nevanlinna:
    """
    Analytic continuation of imaginary-domain data to real-domain data using
    Nevanlinna analytic continuation
    """
    def __init__(self, matsubara_freq, ng):
        """
        Args:
            matsubara_freq: list (purely imaginary) Matsubara frequencies,
                typically of the form (2*pi*n/T) for an integer n in [0, T-1]
                and T the temporal size of the box.
            ng: the values of the Nevanlinna Green's function evaluated at the
                Matsubara frequencies
        """
        imag = ImaginaryDomainData(matsubara_freq, ng)
        self.schur = Schur(imag)

    def __call__(self, start=0, stop=1, num=50, eta=1e-3, map_back=True):
        """
        Computes the values of the Nevanlinna Green's function just above the
        real axis, i.e., along the line "omega + i*eta", where omega is real
        and eta is a small imaginary "regulator" parameter.
        Args:
            start: float, minumum value for omega
            stop: float, maximum value for omega
            num: int, the total number of values for omega in [start, stop]
            eta: real, the value of the regulator
            map_back: bool, whether to map the result for the Nevanlinna Green
                function from the unit disk back to the upper half plane
        Returns:
            omega, result: arrays with the values of "omega + i*eta" and the
                final result for the Nevanlinna Green function.
        """
        omega = RealDomainData(start, stop, num, eta=eta).freq
        result = [self.schur(z, map_back=map_back) for z in omega]
        result = np.array(result)
        return omega, result

