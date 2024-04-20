import numpy as np
import matplotlib.pyplot as plt
import iminuit.util
from iminuit.cost import poisson_chi2

class BBlite_histograms:
    def __init__(self, regions, data, templates):
        """
        regions: list of strings
        data: list of histograms
        templates: list of histograms
        """
        self.regions = regions
        self.data = data
        self.xe = [d_i.axes[0].edges for d_i in data]
        self.templates = templates

    def _pred(self, templates, par):
        """
        templates: list of list of histograms (loop over processes, then ndf and df)
        par: array of k factors for each process + S_decay parameter
        """
        n_processes = len(templates)
        n_bins = templates[0][0].axes[0].size
        if isinstance(par, iminuit.util.ValueView):
            par = np.fromiter(par.to_dict().values(), dtype=float)
        elif not isinstance(par, np.ndarray):
            par = np.array(par)
        if par.ndim != 1:
            par = np.ravel(par)
        k_factors = par[:-1, np.newaxis]
        S_decay = np.array(n_processes * [[[par[-1]**i if i else 0 for i in range(8)]] * n_bins])
        t_ndf = np.array([tt[0].values() for tt in templates])
        vt_ndf = np.array([tt[0].variances() for tt in templates])
        t_df = np.array([tt[1].values() for tt in templates])
        vt_df = np.array([tt[1].variances() for tt in templates])
        t = t_ndf + np.sum(t_df * S_decay, axis=-1)
        vt = vt_ndf + np.sum(vt_df * (S_decay**2), axis=-1)
        mu = np.sum(k_factors * t, axis=-2)
        mu_var = np.sum((k_factors**2) * vt , axis=-2)
        return mu, mu_var

    def _template_chi2_jsc(self, n, mu, mu_var):
        n, mu, mu_var = np.atleast_1d(n, mu, mu_var)
        beta_var = mu_var / mu**2
        p = 0.5 - 0.5 * mu * beta_var
        beta = p + np.sqrt(p**2 + n * beta_var)
        return poisson_chi2(n, mu * beta) + np.sum((beta - 1) ** 2 / beta_var)

    def __call__(self, par):
        """
        par: k factors for each process
        """
        data = self.data
        templates = self.templates
        r = 0 
        for i, _ in enumerate(self.regions):
            mu_i, mu_var_i = self._pred(templates[i], par)
            n_i = data[i].values()
            ma = mu_i > 0
            r += self._template_chi2_jsc(n_i[ma], mu_i[ma], mu_var_i[ma])
        return r

    @property
    def ndata(self):
        n, t = self.data
        return np.prod(n.shape) + np.prod(t.shape)

    def visualize(self, par):
        regions = self.regions
        n = self.data
        templates = self.templates
        xe = self.xe
        fig = plt.gcf()
        # n_regions = len(regions)
        # fig.set_figwidth((n_regions / 2) * fig.get_figwidth() / 1.5)
        # fig.set_figheight((n_regions / 2) * fig.get_figheight() / 1.5)
        # _, ax = plt.subplots(n_regions / 2, n_regions / 2, num=fig.number)
        fig.set_figwidth(4 * fig.get_figwidth() / 1.5)
        fig.set_figheight(2 * fig.get_figheight() / 1.5)
        _, ax = plt.subplots(2, 4, num=fig.number)
        j=0
        for i, r in enumerate(regions):
            if i > 3:
                j = 1
            n_i = n[i].values()
            ne_i = n_i**0.5
            mu_i, mu_var_i = self._pred(templates[i], par)
            cx = 0.5 * (xe[i][1:] + xe[i][:-1])
            # if len(regions) == 1:
            #     ax = [ax]
            # ax[i].errorbar(cx, n_i, ne_i, fmt="ok")
            # for fill in [False, True]:
            #     ax[i].stairs(mu_i + mu_var_i**0.5, xe[i], baseline=mu_i - mu_var_i**0.5, fill=fill, color="C0")
            # ax[i].set_title(r)
            ax[j, i%4].errorbar(cx, n_i, ne_i, fmt="ok")
            for fill in [False, True]:
                ax[j, i%4].stairs(mu_i + mu_var_i**0.5, xe[i], baseline=mu_i - mu_var_i**0.5, fill=fill, color="C0")
            ax[j, i%4].set_title(r)


class BBlite_sources:
    def __init__(self, regions, xe, n, t_ndf, vt_ndf, t_df, vt_df):
        """
        regions: list of strings
        xe: bin edges
        n: observed yields
        t_ndf: expected yields for the non-decays-in-flight
        t_df: expected yields for the decays-in-flight
        """
        self.regions = regions
        self.xe = xe
        self.data = n, t_ndf, vt_ndf, t_df, vt_df

    def _pred(self, t_ndf, vt_ndf, t_df, vt_df, par):
        t_ndf_sum = np.sum(t_ndf, axis=-1, keepdims=True)
        t_df_sum = np.sum(t_df, axis=-1, keepdims=True)
        if isinstance(par, iminuit.util.ValueView):
            par = np.fromiter(par.to_dict().values(), dtype=float)
        elif not isinstance(par, np.ndarray):
            par = np.array(par)
        if par.ndim != 1:
            par = np.ravel(par)
        k_factors = par[:-1, np.newaxis]
        S_decay = par[-1]
        t = t_ndf + t_df * S_decay
        vt = vt_ndf + vt_df * (S_decay**2)
        mu = np.sum(k_factors * t, axis=-2)
        mu_var = np.sum((k_factors**2) * vt , axis=-2)
        return mu, mu_var

    def _template_chi2_jsc(self, n, mu, mu_var):
        n, mu, mu_var = np.atleast_1d(n, mu, mu_var)
        beta_var = mu_var / mu**2
        p = 0.5 - 0.5 * mu * beta_var
        beta = p + np.sqrt(p**2 + n * beta_var)
        return poisson_chi2(n, mu * beta) + np.sum((beta - 1) ** 2 / beta_var)

    def __call__(self, par):
        """
        par: k factors for each process
        """
        n, t_ndf, vt_ndf, t_df, vt_df = self.data
        mu, mu_var = self._pred(t_ndf, vt_ndf, t_df, vt_df, par)
        r = 0 
        for i, _ in enumerate(self.regions):
            n_i = n[i]
            mu_i = mu[i]
            mu_var_i = mu_var[i]
            ma = mu_i > 0
            r += self._template_chi2_jsc(n_i[ma], mu_i[ma], mu_var_i[ma])
        return r

    @property
    def ndata(self):
        n, t = self.data
        return np.prod(n.shape) + np.prod(t.shape)

    def visualize(self, par):
        regions = self.regions
        n, t_ndf, vt_ndf, t_df, vt_df = self.data
        ne = n ** 0.5
        xe = self.xe
        mu, mu_var = self._pred(t_ndf, vt_ndf, t_df, vt_df, par)
        fig = plt.gcf()
        n_regions = len(regions)
        fig.set_figwidth(n_regions * fig.get_figwidth() / 1.5)
        _, ax = plt.subplots(1, n_regions, num=fig.number)
        for i, r in enumerate(regions):
            n_i = n[i]
            mu_i = mu[i]
            ne_i = ne[i]
            mu_var_i = mu_var[i]
            cx = 0.5 * (xe[i][1:] + xe[i][:-1])
            if len(regions) == 1:
                ax = [ax]
            ax[i].errorbar(cx, n_i, ne_i, fmt="ok")
            for fill in [False, True]:
                ax[i].stairs(mu_i + mu_var_i**0.5, xe[i], baseline=mu_i - mu_var_i**0.5, fill=fill, color="C0")
            ax[i].set_title(r)


class BBlite:
    def __init__(self, regions, xe, n, t):
        """
        regions: list of strings
        xe: bin edges
        n: observed yields
        t: expected yields
        """
        self.regions = regions
        self.xe = xe
        self.data = n, t

    def _pred(self, t, par):
        t_sum = np.sum(t, axis=-1, keepdims=True)
        if isinstance(par, iminuit.util.ValueView):
            par = np.fromiter(par.to_dict().values(), dtype=float)
        elif not isinstance(par, np.ndarray):
            par = np.array(par)
        if par.ndim != 1:
            par = np.ravel(par)
        k_factors = par[:, np.newaxis]
        mu = np.sum(k_factors * t, axis=-2)
        mu_var = np.sum((k_factors**2) * t , axis=-2)
        return mu, mu_var

    def _template_chi2_jsc(self, n, mu, mu_var):
        n, mu, mu_var = np.atleast_1d(n, mu, mu_var)
        beta_var = mu_var / mu**2
        p = 0.5 - 0.5 * mu * beta_var
        beta = p + np.sqrt(p**2 + n * beta_var)
        return poisson_chi2(n, mu * beta) + np.sum((beta - 1) ** 2 / beta_var)

    def __call__(self, par):
        """
        par: k factors for each process
        """
        n, t = self.data
        mu, mu_var = self._pred(t, par)
        r = 0 
        for i, _ in enumerate(self.regions):
            n_i = n[i]
            mu_i = mu[i]
            mu_var_i = mu_var[i]
            ma = mu_i > 0
            r += self._template_chi2_jsc(n_i[ma], mu_i[ma], mu_var_i[ma])
        return r

    @property
    def ndata(self):
        n, t = self.data
        return np.prod(n.shape) + np.prod(t.shape)

    def visualize(self, par):
        regions = self.regions
        n, t = self.data
        ne = n ** 0.5
        xe = self.xe
        mu, mu_var = self._pred(t, par)
        fig = plt.gcf()
        n_regions = len(regions)
        fig.set_figwidth(n_regions * fig.get_figwidth() / 1.5)
        _, ax = plt.subplots(1, n_regions, num=fig.number)
        for i, r in enumerate(regions):
            n_i = n[i]
            t_i = t[i]
            mu_i = mu[i]
            ne_i = ne[i]
            mu_var_i = mu_var[i]
            cx = 0.5 * (xe[i][1:] + xe[i][:-1])
            if len(regions) == 1:
                ax = [ax]
            ax[i].errorbar(cx, n_i, ne_i, fmt="ok")
            for fill in [False, True]:
                ax[i].stairs(mu_i + mu_var_i**0.5, xe[i], baseline=mu_i - mu_var_i**0.5, fill=fill, color="C0")
            ax[i].set_title(r)

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
