import numpy as np
import scipy.stats
import typing

class GaussianVariable:
    def __init__(self, mean : float, standard_deviation : float):
        self.mean = mean
        self.standard_deviation = standard_deviation
    
    def __repr__(self):
        return "{:.1e}±{:.1e}".format(self.mean, self.standard_deviation)
    
    def __add__(self, other):
        if isinstance(other, GaussianVariable):
            return GaussianVariable(
                mean = self.mean + other.mean,
                standard_deviation = np.sqrt(
                    np.power(self.standard_deviation, 2) +
                    np.power(other.standard_deviation, 2)
                ),
            )
        else:
            raise NotImplementedError
    
    def __sub__(self, other):
        if isinstance(other, GaussianVariable):
            return GaussianVariable(
                mean = self.mean - other.mean,
                standard_deviation = np.sqrt(
                    np.power(self.standard_deviation, 2) +
                    np.power(other.standard_deviation, 2)
                ),
            )
        else:
            raise NotImplementedError
    
    def __mul__(self, other):
        if isinstance(other, GaussianVariable):
            # This is kind of experimental. TODO find a more reputable approximation and replace
            # TODO add better criteria for good approximation

            self_cv = np.abs(self.standard_deviation / self.mean)
            other_cv = np.abs(other.standard_deviation / other.mean)
            if self_cv > 0 and other_cv > 0:
                assert 1/(1/self_cv+1/other_cv) < 0.15, "Result is not near-normal"

            return GaussianVariable(
                mean = self.mean * other.mean,
                standard_deviation = np.sqrt(
                    np.power(self.mean * other.standard_deviation, 2) +
                    np.power(other.mean * self.standard_deviation, 2)
                ),
            )
        else:
            raise NotImplementedError
    
    def __truediv__(self, other):
        if isinstance(other, GaussianVariable):
            # Approximation from
            #   Díaz-Francés, E., & Rubio, F. J. (2012). On the existence of a normal
            #   approximation to the distribution of the ratio of two independent normal
            #   random variables. Statistical Papers, 54(2), 309–323.
            #   doi:10.1007/s00362-012-0429-2

            # However, GAMMA is kind of made up. Need to better understand and fix.
            # TODO replace gamma definition with more appropriate

            LAMBDA = .4
            self_cv = self.standard_deviation / self.mean
            other_cv = other.standard_deviation / other.mean
            assert self_cv <= LAMBDA, "Result is not near-normal"
            GAMMA = 0.4 * np.sqrt(np.power(LAMBDA, 2) - np.power(self_cv, 2))
            assert other_cv <= GAMMA, "Result is not near-normal"

            return GaussianVariable(
                mean = self.mean / other.mean,
                standard_deviation = np.sqrt(
                    np.power(self.mean / other.mean, 2) *
                    (
                        np.power(self.standard_deviation / self.mean, 2) +
                        np.power(other.standard_deviation / other.mean, 2)
                    )
                ),
            )
        else:
            raise NotImplementedError
    
    def pdf(self, x):
        return scipy.stats.norm.pdf(
            x,
            self.mean,
            self.standard_deviation
        )
    
    def cdf(self, x):
        return scipy.stats.norm.cdf(
            x,
            self.mean,
            self.standard_deviation
        )
    
    def sample(self, n=10000):
        return np.random.normal(
            loc=self.mean,
            scale=self.standard_deviation,
            size=(n,)
        )