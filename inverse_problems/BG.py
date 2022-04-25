"""
BG object combines functions into single class, modelled after Will's "BackusGilbert" object.
Includes functionality for mpmath package for extended precision to avoid errors caused by 
Backus-Gilbert producing large coefficents that oscillate in sign. Intend to add functionality
for correlator data produced with periodic boundary conditions {UNIMPLEMENTED AT THIS TIME}
"""
import gvar as gv
import mpmath as mpm
import copy as copy

class BG(object):
    """
    Creates the BG object
    data: list of gvar variables - the correlator data. If errors are not
        given, covariance matrix evaluates to None or 0 * nxn Identity
    t: list of integers - time-slice values corresponding to the correlator
        data. Cannot include 0.
    lam: float - lambda parameter that provides error weighting in the BG
        algorithm. Defaults to 0, must be in range [0,1] inclusive
    sym: boolean - flag to use PBC version of methods {UNIMPLEMENTED AT THIS
        TIME}. Defaults to false.
    """
    def __init__(self, data, t, lam=0.0, sym=False):
        if 0 in t:
            raise ValueError("Undefined when t=0.")

        self.t = mpm.matrix(t)
        self.data = mpm.matrix(gv.mean(data))
        self.lam = lam
        try:
            cov = gv.evalcov(data)
            self.cov = mpm.matrix(cov)
        except AttributeError:
            self.cov = None
        
    """
    Getters
    *
    *
    """
    def get_t(self):
        return self.t.copy()
    
    def get_data(self):
        return self.data.copy()
    
    def get_cov(self):
        return self.cov.copy()
    
    def get_lam(self):
        return self.lam
    
   
    """
    *
    *
    End getters
    """
    
    """
    Setters
    *
    *
    """
    def set_t(self, t):
        self.t = t
    
    def set_data(self, data):
        self.data = data
    
    def set_cov(self, cov):
        self.cov = cov

    """
    Allows for changing values of BG method tuning parameter lambda, corresponds to
        Will's "set_lambda" method
    lam: float - in range of [0, 1], corresponds to weighing the algorithm
        more (lam = 1) or the errors in the data more (lam = 0)
    """
    def set_lambda(self, lam):
        if (lam >= 0) and (lam <= 1):
            self.lam = lam
        else:
            msg = "Error: lambda must be in the interval [0,1]."
            raise ValueError(msg)
    
    """
    *
    *
    End setters
    """

    """
    Response kernel, corresponds to "basis" function or Will's "r" method
    E: float - energy at which the kernel is evaluated
    """
    def r(self, E):
        t = self.get_t()
        out = self.get_t()
        for i in range(out.rows):
            out[i] = mpm.exp(-E*out[i])
        return out
    
    """
    Integrated response kernel, corresponds to "R" function or Will's "R" method
    """
    def R(self):
        t = self.get_t()
        out = self.get_t()
        for i in range(out.rows):
            out[i] = mpm.fdiv(1, t[i])
        return out
    
    """
    Spread matrix, corresponds to "contruct_A" function or Will's "w" method
    E: float - energy at which kernel is evaluated
    """
    def A(self, E):
        t = self.get_t()
        size = t.rows
        A = mpm.matrix(size, size)

        for i in range(size):
            for j in range(size):
                t_sum = t[i] + t[j]
                try:
                    # Expressions for the Laplace kernel only
                    A[i, j] = (2 + E * (t_sum) * (-2 + E * (t_sum))) / mpm.power(t_sum, 3)
                except ZeroDivisionError:
                    raise ValueError("Error: spreak matrix requires positive t.")
        return A
    
    """
    Functional used in calculating inverse response kernel, corresponds to
        "construct_W" function or part of Will's "q" method
    E: float - energy at which kernel is evaluated
    """
    def W(self, E):
        cov = self.get_cov()
        A = self.A(E)
        lam = self.get_lam()
        if (self.get_lam() > 0.0 and cov is not None):
            W = (1 - lam) * A + lam * cov
            #W = A + lam * cov
        else:
            W = A
        return W
    
    """
    Inverse response kernel, corresponds to "g_coeffs" function or Will's "q" method
    E: float - energy at which kernel is evaluated
    """
    def g(self, E):
        W = self.W(E)
        R = self.R()
        
        W_inv = W**-1
        numer = W_inv * R
        denom = (R.T * W_inv * R)[0]
        try:
            return numer / denom
        except ZeroDivisionError:
            msg = "Error: failed to compute inverse response kernel."
            raise ValueError(msg)
            
    """
    The averaging kernel delta(E, E_star).
    E: float - energy spectrum
    E_star: float - energy value at which our delta function peaks
    """
    def delta(self, E, E_star):
        g = self.g(E)
        r = self.r(E_star)
        return (r.T * g)[0]
    
    """
    The solution of inverse laplace transform, corresponds to g_coeffs * correlator
        data or Will's "estimator" method
    """
    def reconstruct(self, E):
        data = self.get_data()
        g = self.g(E)
        
        return (data.T * g)[0]
    
    """
    Variance of the reconstructed spectral function based on input errors,
        corresponds to Will's "variance" method
    E: float - energy at which kernel is evaluated
    """
    def variance(self, E):
        if self.cov is None:
            msg = "Error no covariance matrix specified."
            raise ValueError(msg)
        cov = self.get_cov()
        g = self.g(E)
        return (g.T * cov * g)[0]
    
    """
    Square root of the variance, corresponds to Will's "error" method
    E: float - energy at which kernel is evaluated
    """
    def error(self, E):
        var = self.variance(E)
        return mpm.sqrt(var)
    
    """
    Gives approximated spectral function for the input energy spectra, corresponds
        to "g_coeffs * correlator data" for various input energies or Will's
        "solve" method
    E: float - energy spectra
    """
    def solve(self, E):
        avgs = [self.reconstruct(E_star) for E_star in E]
        errs = [self.error(E_star) for E_star in E]
        
        return gv.gvar(avgs, errs)