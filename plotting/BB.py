import numpy as np
import matplotlib.pyplot as plt
from iminuit.cost import poisson_chi2

class BB:
    def __init__(self, xe, n, t):
        self.xe = xe
        self.data = n, t

    def _pred(self, par):
        bins = len(self.xe) - 1
        yields = par[:2]
        nuisances = np.array(par[2:])
        b = nuisances[:bins]
        s = nuisances[bins:]
        mu = 0
        for y, c in zip(yields, (b, s)):
            mu += y * np.array(c) / np.sum(c)
        return mu, b, s

    def __call__(self, par):
        n, t = self.data
        mu, b, s = self._pred(par)
        r = poisson_chi2(n, mu) + poisson_chi2(t[0], b) + poisson_chi2(t[1], s)
        return r

    @property
    def ndata(self):
        n, t = self.data
        return np.prod(n.shape) + np.prod(t.shape)

    def visualize(self, args):
        n, t = self.data
        ne = n ** 0.5
        xe = self.xe
        cx = 0.5 * (xe[1:] + xe[:-1])
        plt.errorbar(cx, n, ne, fmt="ok")
        mu = 0
        mu_var = 0
        for y, c in zip(args[:2], t):
            f = 1 / np.sum(c)
            mu += c * y * f
            mu_var += c * (f * y) ** 2
        mu_err = mu_var ** 0.5
        plt.stairs(mu + mu_err, xe, baseline=mu - mu_err, fill=True, color="C0")
