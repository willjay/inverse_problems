"""
Hansen, Lupo, and Tantalo claim a modification of the Backus-Gilbert method
    used for determining the inverse of the Laplace transform (Hansen, et al.;
    arxiv:1903.06476) that is able to more reliably extract statistics from 
    input correlator data by using a target smering function as the input.
"""
class ModifiedBG(object):
    
    def __init__(self, data, t, E_0, sigma=0.0, lam=0.0, sym=False):
        self.t = mpm.matrix(t)
        self.data = mpm.matrix(gv.mean(data))
        self.lam = mpm.mpf(lam)
        self.sigma = mpm.mpf(sigma)
        self.E_0 = mpm.mpf(E_0)
        self.T = self.t.rows
        self.sym = sym
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
    
    def get_T(self):
        return self.T
    
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
    
    def get_sym(self):
        return self.sym
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
        self.T = self.t.rows
    
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
            self.lam = mpm.mpf(lam)
        else:
            msg = "Error: lambda must be in the interval [0,1]."
            raise ValueError(msg)
            
    def set_sigma(self, sigma):
        self.sigma = mpm.mpf(sigma)
    
    def set_E_0(self, E_0):
        self.E_0 = mpm.mpf(E_0)
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
        const = mpm.fdiv(1, mpm.fmul(mpm.sqrt(mpm.fmul(2,mpm.pi)), mpm.fmul(sigma, Z)))
        arg = (-1*mpm.fmul(mpm.fsub(E, E_star),mpm.fsub(E, E_star)))
        arg = mpm.fdiv(arg, mpm.fmul(2, mpm.fmul(sigma, sigma)))
        
        return mpm.fmul(const, mpm.exp(arg))
    
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
        sigma = self.get_sigma()
        arg = mpm.fdiv(E_star, mpm.fmul(sigma, mpm.sqrt(2)))
        return mpm.fdiv(mpm.fadd(1, mpm.erf(arg)), 2)
    
    def N(self, E_star):
        t = self.get_t()
        lam = self.get_lam()
        sigma = self.get_sigma()
        out = self.get_t()
        Z = self.Z(E_star)
        
        coef = mpm.fdiv(mpm.fsub(1, lam), mpm.fdiv(2, Z))
        for i in range(out.rows):
            arg = mpm.fmul(2, E_star)
            arg = mpm.fsub(arg, mpm.fmul(t[i], mpm.fmul(sigma, sigma)))
            arg = mpm.fdiv(arg, 2)
            arg = mpm.fmul(arg, -t[i])
            #arg = -t[i] * (2 * E_star - t[i] * sigma**2) / 2
            out[i] = coef * mpm.exp(arg)
        return out
    
    def F(self, E_star):
        t = self.get_t()
        sigma = self.get_sigma()
        E_0 = self.get_E_0()
        out = self.get_t()
        
        for i in range(out.rows):
            arg = mpm.fmul(-t[i], mpm.fmul(sigma, sigma))
            arg = mpm.fadd(arg, E_star)
            arg = mpm.fsub(arg, E_0)
            arg = mpm.fdiv(arg, mpm.fmul(mpm.sqrt(2),sigma))
            #arg = (-t[i] * sigma ** 2 + E_star - E_0) / (mpm.sqrt(2) * sigma)
            out[i] = mpm.fadd(1, mpm.erf(arg))
        return out
    
    def f(self, E_star):
        t_max = self.t_max()
        T = self.get_T()
        sym = self.get_sym()
        
        N = self.N(E_star)
        F = self.F(E_star)
        out = self.get_t()[:t_max]
        for i in range(t_max):
            out[i] = mpm.fmul(N[i + 1], F[i + 1])
            if sym:
                out[i] = mpm.fadd(out[i], mpm.fmul(N[T - i - 1], F[T - i - 1]))
                
        return out
    
    """
    *
    *
    End auxillary functions
    """
    
    """
    Used for determining t_max as defined in HLT
    """
    def t_max(self):
        T = self.get_T()
        sym = self.get_sym()
        if sym:
            return int(T / 2)
        return T - 1
    
    """
    Response kernel, corresponds to "basis" function or Will's "r" method
    E: float - energy at which the kernel is evaluated
    """
    def r(self, E):
        t = self.get_t()
        T = self.get_T()
        sym = self.get_sym()
        t_max = self.t_max()
        
        out = self.get_t()[:t_max]
        
        for i in range(t_max):
            out[i] = mpm.exp(mpm.fmul(-E,t[i + 1]))
            if sym:
                out[i] += mpm.exp(mpm.fmul(-E,mpm.fsub(T, t[i + 1])))
        return out
    
    """
    Integrated response kernel, corresponds to "R" function or Will's "R" method
    """
    def R(self):
        t = self.get_t()
        T = self.get_T()
        sym = self.get_sym()
        t_max = self.t_max()
        
        out = self.get_t()[:t_max]
        for i in range(t_max):
            out[i] = mpm.fdiv(1, (t[i + 1]))
            if sym:
                
                out[i] += mpm.fdiv(1, mpm.fsub(T, t[i + 1]))
        return out
    
    """
    Spread matrix, corresponds to "contruct_A" function or Will's "w" method
    E: float - energy at which kernel is evaluated
    """
    def A(self):
        t = self.get_t()
        T = self.get_T()
        E_0 = self.get_E_0()
        t_max = self.t_max()
        sym = self.get_sym()
        
        A = mpm.matrix(t_max, t_max)

        for i in range(t_max):
            for j in range(t_max):
                t_sum = mpm.fadd(t[i + 1], t[j + 1])
                #t_dif = mpm.fsub(t[i + 1], t[j + 1])
                try:
                    # Expressions for the Laplace kernel only
                    A[i, j] = mpm.fdiv(mpm.exp(mpm.fmul(-E_0,t_sum)), t_sum)
                    if sym:
                        A[i, j] += mpm.fdiv(mpm.exp(-E_0*(T - t[j + 1] + t[i + 1])), T - t[j + 1] + t[i + 1]) + \
                            mpm.fdiv(mpm.exp(-E_0*(T - t[i + 1] + t[j + 1])), T - t[i + 1] + t[j + 1]) + \
                            mpm.fdiv(mpm.exp(-E_0*(2 * T - t[j + 1] - t[i + 1])), 2 * T - t[j + 1] - t[i + 1])
                except ZeroDivisionError:
                    raise ValueError("Error: spreak matrix requires positive t.")
        return A
    
    """
    Functional used in calculating inverse response kernel, corresponds to
        "construct_W" function or part of Will's "q" method
    E: float - energy at which kernel is evaluated
    """
    def W(self, E):
        data = self.get_data()
        t_max = self.t_max()
        cov = self.get_cov()
        E_0 = self.get_E_0()
        A = self.A()
        lam = self.get_lam()
        if (self.get_lam() > 0.0 and cov is not None):
            #print(A.rows, A.cols)
            #print(cov.rows, cov.cols)
            A_part = mpm.fmul(mpm.fsub(1, lam), A[:t_max,:t_max])
            B_part = mpm.div(mpm.fmul(lam, cov[:t_max,:t_max]), mpm.fmul(data[0], data[0]))
            W = mpm.fadd(A_part, B_part)
            #W = A + lam * cov
        else:
            W = A[:t_max,:t_max]
        return W
    
    """
    Inverse response kernel
    E: float - energy at which the kernel is evaluated
    """
    def g(self, E):
        t_max = self.t_max()
        W = self.W(E)#[:t_max,:t_max]
        f = self.f(E)#[:t_max]
        R = self.R()#[:t_max]
        
        #print(W.rows, f.rows, R.rows)
        
        Winv = W**-1
        denom = (R.T*Winv*R)[0]
        numer = mpm.fsub(1, (R.T * Winv * f)[0])
        return Winv * f + Winv * R * (numer / denom)
    
    """
    Functional used in evaluating the optimal choice of the lambda parameter
    g: mpmath matrix object - output coefficients of the method
    """
    def B_functional(g):
        t_max = self.t_max()
        cov = self.get_cov()[:t_max][:t_max]
        return (g.T * cov * g)[0]
    
    """
    The solution of inverse laplace transform, corresponds to g_coeffs * correlator
        data or Will's "estimator" method
    E: float - energy at which the kernel is evaluated
    """
    def reconstruct(self, E):
        t_max = self.t_max()
        data = self.get_data()[1:t_max+1]
        g = self.g(E)[:t_max]
        
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
        ##print(avgs)
        ##print(errs)
        return gv.gvar(avgs, errs)
