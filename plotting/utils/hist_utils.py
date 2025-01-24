import numpy as np
import hist
import boost_histogram as bh

def slice_hist2d(hist, regions_list, slice_var="y"):
    """
    Inputs:
        hist: 2d Hist histogram.
        regions_list: list of regions using Hist slicing. e.g. [[10j,20j],[20j,30j],...]
        slice_var: 'x' or 'y', which dimensions to slice in
    Returns:
        A list of Hist histograms.
    """
    hist_list = []
    for regions in regions_list:
        if slice_var == "y":
            h = hist[:, regions[0] : regions[1] : sum]
        elif slice_var == "x":
            h = hist[regions[0] : regions[1] : sum, :]
        hist_list.append(h)
    return hist_list

def rebin_piecewise(h_in, bins, histtype="hist"):
    """
    Inputs:
        h : histogram
        bins: list of bins as real numbers
        histtype: one of allowed_histtypes to return

    Returns:
        h_out: a histogram of type 'histtype', rebinned according to desired bins
    """

    # only 1D hists supported for now
    if len(h_in.shape) != 1:
        raise Exception("Only 1D hists supported for now")

    # only hist and bh supported
    allowed_histtypes = ["hist", "bh"]
    if histtype not in allowed_histtypes:
        raise Exception("histtype in not in allowed_histtypes")

    # check that the bins are real numbers
    if any([x.imag != 0 for x in bins]):
        raise Exception("Only pass real-valued bins")

    # split the histogram by the bins
    # and for each bin, calculate total amount of events and variance
    z_vals, z_vars = [], []
    for iBin in range(len(bins) - 1):
        if histtype == "hist":
            bin_lo = bins[iBin] * 1.0j
            bin_hi = bins[iBin + 1] * 1.0j
        elif histtype == "bh":
            bin_lo = bh.loc(bins[iBin])
            bin_hi = bh.loc(bins[iBin + 1])

        h_fragment = h_in[bin_lo:bin_hi]
        z_vals.append(h_fragment.sum().value)
        z_vars.append(h_fragment.sum().variance)

    # fill the histograms
    if histtype == "hist":
        h_out = hist.Hist(
            hist.axis.Variable(bins, label=h_in.axes[0].label, name=h_in.axes[0].name),
            storage=hist.storage.Weight(),
            label=h_in.axes[0].label,
        )
        h_out[:] = np.stack([z_vals, z_vars], axis=-1)

    elif histtype == "bh":
        h_out = bh.Histogram(bh.axis.Variable(bins), storage=bh.storage.Weight())
        h_out[:] = np.stack([z_vals, z_vars], axis=-1)

    return h_out

def hist_mean(hist):
    """
    Calculates the mean of a 1-dimensional Hist histogram.
    """
    bin_values = hist.values()
    bin_centers = hist.axes[0].centers
    mean = np.average(bin_centers, weights=bin_values)
    return mean


def hist_std_dev(hist, axis=0):
    """
    Calculates the standard deviation of a 1-dimensional Hist histogram.
    """
    bin_values = hist.values()
    bin_centers = hist.axes[0].centers
    mean = hist_mean(hist)

    # Calculate the sum of squared differences from the mean
    squared_diff_sum = np.sum(bin_values * (bin_centers - mean) ** 2)

    # Calculate the standard deviation
    standard_deviation = np.sqrt(squared_diff_sum / np.sum(bin_values))

    return standard_deviation

def hist2d_correlation(h):
    """
    Calculates Pearson Coefficient from a 2-dimensional Hist histogram.
    """

    coeff = 0

    assert len(h.axes) == 2

    xvals = h.axes[0].centers
    yvals = h.axes[1].centers
    zvals = h.values()

    xmean = hist_mean(h[:, sum])
    ymean = hist_mean(h[sum, :])
    xdev = hist_std_dev(h[:, sum])
    ydev = hist_std_dev(h[sum, :])

    if xdev == 0 or ydev == 0:
        return

    for i in range(len(xvals)):
        for j in range(len(yvals)):
            coeff += (xvals[i] - xmean) * (yvals[j] - ymean) * zvals[i, j]

    coeff /= xdev * ydev * h.sum().value
    error = np.sqrt((1 - coeff**2) / (h.sum().value - 2))

    return coeff, error


def poly_fit_hist2d(h, deg=1):
    z_values = h.values().flatten()
    x_centers = h.axes[0].centers
    y_centers = h.axes[1].centers
    x_values = np.array([])
    y_values = np.array([])
    for i in range(len(x_centers)):
        x_values = np.concatenate((x_values, np.ones_like(y_centers) * x_centers[i]))
    for _i in range(len(x_centers)):
        y_values = np.concatenate((y_values, y_centers))
    p = np.poly1d(np.polyfit(x_values, y_values, deg, w=z_values, cov=False))
    logging.info("Linear fit result:", p)
    return p