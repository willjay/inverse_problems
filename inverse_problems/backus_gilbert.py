"""
Code for Backus-Gilbert reconstruction of the inverse Laplace transform.

The response kernel is Laplace kernel, r_i(nu) = r(nu, t_i) = exp(-nu*t_i).

The integrated response kernel is the integral of the response kernal with
respect to nu R_i = R(t_i) = \int_0^\infty d\nu r(\nu, t_i) = 1/t_i.

The inverse response kernel is
q_i(nu) = q(nu, t_i)
        = Winv(nu)_{ij} R_j / R_k Winv(nu)_{kl} R_l

The inverse response kernel depends is dependes on the kernel and the times t_i.
At this state, it does not depend at all on the data. The matrix function
Winv(nu)_{ij} is the inverse of the spread matrix
W(nu)_ij = \int_0^\infty d(nuprime) (nu-nuprime)^2 r_i(nuprime) r_j(nuprime).
For the Laplace kernel, this integral can be evaulated exactly:
W(nu)_ij = nu^2 / (t_i + t_j) - 2*nu / (t_i + t_j)^2 + 2 / (t_i + t_j)^3
"""
import numpy as np
import gvar as gv

class BackusGilbert(object):
    def __init__(self, data, t, lam=0.0):
        if 0 in t:
            raise ValueError("Undefined when t=0.")

        self.t = np.asarray(t,dtype=float)
        self.data = np.asarray(gv.mean(data))
        self.lam = lam

        try:
            self.cov = gv.evalcov(data)
        except AttributeError:
            self.cov = None

    def set_lambda(self, lam):
        if (lam >= 0) and (lam <= 1):
            self.lam = lam
        else:
            msg = "Error: lambda must be in the interval [0,1]."
            raise ValueError(msg)

    def r(self, nu):
        """
        Response kernel (the Laplace kernel)
        Compare to Eq (19.6.3) in Numerical Recipes, 3rd Edition, p 1014
        """
        return np.exp(-nu*self.t)

    def R(self):
        """
        Integrated response kernel for the Laplace kernel
        Compare to Eq (19.6.5) in Numerical Recipes, 3rd Edition, p 1015
        """
        return 1/self.t

    def q(self, nu):
        """
        Inverse response kernel q_i(nu) = q(nu; t_i)
        Compare to Eq (19.6.12) in Numerical Recipes, 3rd Edition, p 1015
        """
        w = self.w(nu) # W_{ij}, the spread matrix
        R = self.R()   # R_i, the integrated kernel

        if (self.lam > 0) and (self.cov is not None):
            a = w + self.lam*self.cov
        else:
            a = w
        # Solve a*x = b for x: x = a_{-1}*b
        # W(nu)^{-1}_{ij} R_j
        winv_R = np.linalg.solve(a=a, b=R)

        # The denominator
        R_winv_R = np.dot(R, winv_R)
        try:
            q = winv_R / R_winv_R
        except ZeroDivisionError:
            msg = "Error: failed to compute inverse response kernel."
            raise ValueError(msg)
        return q

    def w(self,nu):
        """
        The spread matrix W(mu)_{ij} for the Laplace kernel.
        Compare to Eq (19.6.10) in Numerical Recipes, 3rd Edition, p 1015
        The integral has been evaluated explicitly for the Laplace kernel.
        """
        t = self.t
        size = len(t)
        w = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                try:
                    # Expressions for the Laplace kernel only
                    ti_plus_tj = t[i] + t[j]
                    w[i,j] = nu**2.0 / ti_plus_tj\
                             - 2.0*nu / ti_plus_tj**2.0\
                             + 2.0 / ti_plus_tj**3.0
                except ZeroDivisionError:
                    raise ValueError("Error: spreak matrix requires positive t.")
        return w

    def delta(self, nu, nuprime):
        """
        The averaging kernel delta(nu, nuprime).
        Compare to Eq (19.6.6) in Numerical Recipes, 3rd Edition, p 1015
        """
        q = self.q(nu)      # q_i(nu)
        r = self.r(nuprime) # r_i(nuprime)
        return np.dot(r,q)

    def estimator(self, nu):
        """
        The estimator -- the solution to the inverse Laplace transform.
        Compare to Eq (19.6.13) in Numerical Recipes, 3rd Edition, p 1015
        """
        data = self.data
        q = self.q(nu)
        return np.dot(data, q)

    def variance(self, nu):
        """
        Variance in the estimator from standard propagation of errors.
        Compare to Eq (19.6.8) in Numerical Recipes, 3rd Edition, p 1015
        """
        if self.cov is None:
            msg = "Error no covariance matrix specified."
            raise ValueError(msg)
        s = self.cov
        q = self.q(nu)
        return np.einsum("i,ij,j", q, s, q)

    def error(self, nu):
        """
        Error in the estimator from standard propagation of errors.
        As usual, the variance is the square of the error.
        """
        var = self.variance(nu)
        return np.sqrt(var)

    def solve(self, nus):
        """
        Solves the inverse problem for several values of nu.
        """
        y = [self.estimator(nu) for nu in nus]
        yerr = [self.error(nu) for nu in nus]
        return gv.gvar(y,yerr)



