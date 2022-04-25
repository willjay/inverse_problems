"""
Hansen, Lupo, and Tantalo claim a modification of the Backus-Gilbert method
    used for determining the inverse of the Laplace transform (Hansen, et al.;
    arxiv:1903.06476) that is able to more reliably extract statistics from 
    input correlator data by using a target smering function as the input.

    This will probably be replaced by a more-faithful-to-the-paper version to ensure
        everything is working properly
"""
class ModifiedBG(object):
    
    def __init__(self, data, t, E_0, sigma=0.0, lam=0.0, sym=False):
        if 0 in t:
            raise ValueError("Undefined when t=0.")

        self.t = mpm.matrix(t)
        self.data = mpm.matrix(gv.mean(data))
        self.lam = lam
        self.sigma = sigma
        self.E_0 = E_0
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
    
    def get_sigma(self):
        return self.sigma
    
    def get_E_0(self):
        return self.E_0
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
        self.t = mpm.matrix(t)
    
    def set_data(self, data):
        self.data = mpm.matrix(gv.mean(data))
        try:
            cov = gv.evalcov(data)
            self.cov = mpm.matrix(cov)
        except AttributeError:
            self.cov = None
    
    def set_cov(self, cov):
        self.cov = mpm.matrix(cov)

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
            
    def set_sigma(self, sigma):
        self.sigma = sigma
    
    def set_E_0(self, E_0):
        self.E_0 = E_0
    """
    *
    *
    End setters
    """
    
    """
    The target smearing function used as an input to the method.
    E: float - energy spectra
    E_star: float - energy at which the smearing function peaks
    """
    def target(self, E, E_star):
        sigma = self.get_sigma()
        Z = self.Z(E_star)
        const = 1 / (mpm.sqrt(2) * sigma * Z)
        arg = (-(E - E_star)**2) / (2 * sigma**2)
        
        return const * mpm.exp(arg)
    
    def delta_bar(self, E, E_star):
        g = self.g(E_star)
        r = self.r(E)
        
        return (g.T * r)[0]
        
    
    """
    Auxillary functions
    *
    *
    """
    def Z(self, E_star):
        sigma = self.sigma
        arg = E_star / (sigma * mpm.sqrt(2))
        return 0.5 * (1 + mpm.erf(arg))
    
    def N(self, E_star):
        t = self.get_t()
        lam = self.get_lam()
        sigma = self.get_sigma()
        out = self.get_t()
        
        coef = (1 - lam) / (2 * self.Z(E_star))
        for i in range(out.rows):
            exponent = -t[i] * (2 * E_star - t[i] * sigma**2) / 2
            out[i] = coef * mpm.exp(exponent)
        return out
    
    def F(self, E_star):
        t = self.get_t()
        sigma = self.get_sigma()
        E_0 = self.get_E_0()
        out = self.get_t()
        
        for i in range(out.rows):
            arg = (-t[i] * sigma ** 2 + E_star - E_0) / (mpm.sqrt(2) * sigma)
            out[i] = 1 + mpm.erf(arg)
        return out
    
    def f(self, E_star):
        N = self.N(E_star)
        F = self.F(E_star)
        out = self.get_t()
        for i in range(out.rows):
            out[i] = N[i] * F[i]
        return out
    
    """
    *
    *
    End auxillary functions
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
    def A(self):
        t = self.get_t()
        E_0 = self.get_E_0()
        size = t.rows
        A = mpm.matrix(size, size)

        for i in range(size):
            for j in range(size):
                t_sum = t[i] + t[j]
                try:
                    # Expressions for the Laplace kernel only
                    A[i, j] = mpm.fdiv(mpm.exp(-E_0*t_sum), t_sum)
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
        E_0 = self.get_E_0()
        A = self.A()
        lam = self.get_lam()
        if (self.get_lam() > 0.0 and cov is not None):
            W = (1 - lam) * A + lam * cov
            #W = A + lam * cov
        else:
            W = A
        return W
    
    """
    Inverse response kernel
    E: float - energy at which the kernel is evaluated
    """
    def g(self, E):
        W = self.W(E)
        f = self.f(E)
        R = self.R()
        
        Winv = W**-1
        denom = (R.T*Winv*R)[0]
        numer = 1 - (R.T * Winv * f)
        return Winv * f + Winv * R * (numer / denom)
    
    """
    Functional used in evaluating the optimal choice of the lambda parameter
    g: mpmath matrix object - output coefficients of the method
    """
    def B_functional(g):
        cov = self.get_cov()
        return (g.T * cov * g)[0]
    
    """
    The solution of inverse laplace transform, corresponds to g_coeffs * correlator
        data or Will's "estimator" method
    E: float - energy at which the kernel is evaluated
    """
    def reconstruct(self, E):
        data = self.get_data()
        g = self.g(E)
        
        return (data.T * g)[0]
    
    """
    Quantity used to determine the bias of the method
    E: float - energy spectrum
    E_star: float - energy in the neighborhood of interest in our spectra
    """
    def delta(self, E, E_star):
        target = self.target(E, E_star)
        delta_bar = self.delta_bar(E, E_star)
        
        return 1 - (delta_bar/target)
    
    """
    Gives an estimate of the systematic uncertainty of the modified method.
    E: float - energy spectra
    """
    def syst_uncertainty(self, E):
        delta = abs(self.delta(E, E))
        reconstruct = self.reconstruct(E)
        return delta * reconstruct
    
    """
    IN PROGRESS!
    HLT suggest a method for determining statistical errors
    TODO: Actually get this working, this isn't playing well unless the errors
        are just astronomical
    """
    def uncertainty(self, E):
        """
        cov = self.get_cov()
        g = self.g(E)
        syst_uncertainty = self.syst_uncertainty(E)
        #stat_uncertainty = (g.T * cov * g)[0]
        stat_uncertainty = []
        for i in range(cov.rows):
            stat_uncertainty.append(cov[i,i])
        
        tot_uncertainty = []
        for i in range(cov.rows):
            tot_uncertainty.append(mpm.sqrt((stat_uncertainty[i])**2 + (0.68 * syst_uncertainty)**2))
        #tot_uncertainty = mpm.sqrt((stat_uncertainty)**2 + (0.68 * syst_uncertainty)**2)
        """
        tot_uncertainty = 0.68 * self.syst_uncertainty(E)
        return tot_uncertainty
    

    """
    IN PROGRESS!
    TODO: Fix the uncertainty() method
    """
    def error(self, E):
        uncertainty = self.uncertainty(E)
        return mpm.sqrt(abs(uncertainty))
        #out = []
        #for i in uncertainty:
        #    out.append(mpm.sqrt(abs(i)))
        #return out
    
    def solve(self, E):
        avgs = [self.reconstruct(E_star) for E_star in E]
        errs = [self.error(E_star) for E_star in E]
        #print(avgs)
        #print(errs)
        return gv.gvar(avgs, errs)
