import sympy
from sympy import diff, symbols
from utils.hist_utils import rebin_piecewise


def make_ABCD_4regions(hist_abcd, xregions, yregions, sum_var=None):
    if sum_var is not None and sum_var not in ["x", "y"]:
        raise ValueError("sum_var must be 'x' or 'y'")
    if sum_var is None:
        A = hist_abcd[xregions[0][0] : xregions[0][1], yregions[0][0] : yregions[0][1]]
        B = hist_abcd[xregions[0][0] : xregions[0][1], yregions[1][0] : yregions[1][1]]
        C = hist_abcd[xregions[1][0] : xregions[1][1], yregions[0][0] : yregions[0][1]]
        SR = hist_abcd[xregions[1][0] : xregions[1][1], yregions[1][0] : yregions[1][1]]
    elif sum_var == "x":
        A = hist_abcd[
            xregions[0][0] : xregions[0][1] : sum, yregions[0][0] : yregions[0][1]
        ]
        B = hist_abcd[
            xregions[0][0] : xregions[0][1] : sum, yregions[1][0] : yregions[1][1]
        ]
        C = hist_abcd[
            xregions[1][0] : xregions[1][1] : sum, yregions[0][0] : yregions[0][1]
        ]
        SR = hist_abcd[
            xregions[1][0] : xregions[1][1] : sum, yregions[1][0] : yregions[1][1]
        ]
    elif sum_var == "y":
        A = hist_abcd[
            xregions[0][0] : xregions[0][1], yregions[0][0] : yregions[0][1] : sum
        ]
        B = hist_abcd[
            xregions[0][0] : xregions[0][1], yregions[1][0] : yregions[1][1] : sum
        ]
        C = hist_abcd[
            xregions[1][0] : xregions[1][1], yregions[0][0] : yregions[0][1] : sum
        ]
        SR = hist_abcd[
            xregions[1][0] : xregions[1][1], yregions[1][0] : yregions[1][1] : sum
        ]

    return A, B, C, SR


def ABCD_4regions_errorProp(hist_abcd, xregions, yregions, sum_var="x", new_bins=None):
    """
    Does the ABCD method for 4 regions, with error propagation.
    """

    if sum_var not in ["x", "y"]:
        raise ValueError("sum_var must be 'x' or 'y'")

    A, B, C, SR = make_ABCD_4regions(hist_abcd, xregions, yregions, sum_var=sum_var)

    # define the histogram that will be scaled up, and the dimension that will be summed
    if sum_var == "x":
        hNUM = B
        hDEN = C
        SR = SR
    elif sum_var == "y":
        hNUM = C
        hDEN = B
        SR = SR

    # rebin
    if new_bins is not None:
        hNUM = rebin_piecewise(hNUM, new_bins)
        SR = rebin_piecewise(SR, new_bins)

    # initialize the SR_exp as empty
    SR_exp = SR.copy()
    SR_exp.view().variance = [0] * len(SR.values())
    SR_exp.view().value = [0] * len(SR.values())

    preds, preds_err = [], []
    for i in range(len(hNUM.values())):
        hNUM_bin = hNUM[i]

        # define the scaling factor function
        a, hnum_bin, hden = symbols(
            "A hnum_bin hden",
        )
        exp = hnum_bin * hden * a**-1

        # defines lists of variables (sympy symbols) and accumulators (hist.sum())
        variables = [a, hnum_bin, hden]
        accs = [
            A.sum(),
            hNUM_bin,
            hDEN.sum(),
        ]

        # calculate scaling factor by substituting values of the histograms' sums for the sympy symbols
        alpha = exp.copy()
        for var, acc in zip(variables, accs):
            alpha = alpha.subs(var, acc.value)

        # calculate the error on the scaling factor
        variance = 0
        for var, acc in zip(variables, accs):
            der = diff(exp, var)
            var = abs(acc.variance)
            variance += der**2 * var
        for var, acc in zip(variables, accs):
            variance = variance.subs(var, acc.value)
        sigma_alpha = variance

        if type(alpha) != sympy.core.numbers.Float or alpha <= 0:
            alpha = 0

        preds.append(alpha)
        preds_err.append(sigma_alpha)

    SR_exp.view().variance = preds_err
    SR_exp.view().value = preds

    return SR, SR_exp


def make_ABCD_6regions(hist_abcd, xregions, yregions, sum_var=None):
    if sum_var is not None and sum_var not in ["x", "y"]:
        raise ValueError("sum_var must be 'x' or 'y'")
    if len(xregions) == 2 and len(yregions) == 3:
        if sum_var is None:
            A = hist_abcd[
                xregions[0][0] : xregions[0][1], yregions[0][0] : yregions[0][1]
            ]
            B = hist_abcd[
                xregions[0][0] : xregions[0][1], yregions[1][0] : yregions[1][1]
            ]
            C = hist_abcd[
                xregions[0][0] : xregions[0][1], yregions[2][0] : yregions[2][1]
            ]
            D = hist_abcd[
                xregions[1][0] : xregions[1][1], yregions[0][0] : yregions[0][1]
            ]
            E = hist_abcd[
                xregions[1][0] : xregions[1][1], yregions[1][0] : yregions[1][1]
            ]
            SR = hist_abcd[
                xregions[1][0] : xregions[1][1], yregions[2][0] : yregions[2][1]
            ]
        elif sum_var == "x":
            A = hist_abcd[
                xregions[0][0] : xregions[0][1] : sum, yregions[0][0] : yregions[0][1]
            ]
            B = hist_abcd[
                xregions[0][0] : xregions[0][1] : sum, yregions[1][0] : yregions[1][1]
            ]
            C = hist_abcd[
                xregions[0][0] : xregions[0][1] : sum, yregions[2][0] : yregions[2][1]
            ]
            D = hist_abcd[
                xregions[1][0] : xregions[1][1] : sum, yregions[0][0] : yregions[0][1]
            ]
            E = hist_abcd[
                xregions[1][0] : xregions[1][1] : sum, yregions[1][0] : yregions[1][1]
            ]
            SR = hist_abcd[
                xregions[1][0] : xregions[1][1] : sum, yregions[2][0] : yregions[2][1]
            ]
        elif sum_var == "y":
            A = hist_abcd[
                xregions[0][0] : xregions[0][1], yregions[0][0] : yregions[0][1] : sum
            ]
            B = hist_abcd[
                xregions[0][0] : xregions[0][1], yregions[1][0] : yregions[1][1] : sum
            ]
            C = hist_abcd[
                xregions[0][0] : xregions[0][1], yregions[2][0] : yregions[2][1] : sum
            ]
            D = hist_abcd[
                xregions[1][0] : xregions[1][1], yregions[0][0] : yregions[0][1] : sum
            ]
            E = hist_abcd[
                xregions[1][0] : xregions[1][1], yregions[1][0] : yregions[1][1] : sum
            ]
            SR = hist_abcd[
                xregions[1][0] : xregions[1][1], yregions[2][0] : yregions[2][1] : sum
            ]
    elif len(xregions) == 3 and len(yregions) == 2:
        if sum_var is None:
            A = hist_abcd[
                xregions[0][0] : xregions[0][1], yregions[0][0] : yregions[0][1]
            ]
            B = hist_abcd[
                xregions[1][0] : xregions[1][1], yregions[0][0] : yregions[0][1]
            ]
            C = hist_abcd[
                xregions[2][0] : xregions[2][1], yregions[0][0] : yregions[0][1]
            ]
            D = hist_abcd[
                xregions[0][0] : xregions[0][1], yregions[1][0] : yregions[1][1]
            ]
            E = hist_abcd[
                xregions[1][0] : xregions[1][1], yregions[1][0] : yregions[1][1]
            ]
            SR = hist_abcd[
                xregions[2][0] : xregions[2][1], yregions[1][0] : yregions[1][1]
            ]
        elif sum_var == "x":
            A = hist_abcd[
                xregions[0][0] : xregions[0][1] : sum, yregions[0][0] : yregions[0][1]
            ]
            B = hist_abcd[
                xregions[1][0] : xregions[1][1] : sum, yregions[0][0] : yregions[0][1]
            ]
            C = hist_abcd[
                xregions[2][0] : xregions[2][1] : sum, yregions[0][0] : yregions[0][1]
            ]
            D = hist_abcd[
                xregions[0][0] : xregions[0][1] : sum, yregions[1][0] : yregions[1][1]
            ]
            E = hist_abcd[
                xregions[1][0] : xregions[1][1] : sum, yregions[1][0] : yregions[1][1]
            ]
            SR = hist_abcd[
                xregions[2][0] : xregions[2][1] : sum, yregions[1][0] : yregions[1][1]
            ]
        elif sum_var == "y":
            A = hist_abcd[
                xregions[0][0] : xregions[0][1], yregions[0][0] : yregions[0][1] : sum
            ]
            B = hist_abcd[
                xregions[1][0] : xregions[1][1], yregions[0][0] : yregions[0][1] : sum
            ]
            C = hist_abcd[
                xregions[2][0] : xregions[2][1], yregions[0][0] : yregions[0][1] : sum
            ]
            D = hist_abcd[
                xregions[0][0] : xregions[0][1], yregions[1][0] : yregions[1][1] : sum
            ]
            E = hist_abcd[
                xregions[1][0] : xregions[1][1], yregions[1][0] : yregions[1][1] : sum
            ]
            SR = hist_abcd[
                xregions[2][0] : xregions[2][1], yregions[1][0] : yregions[1][1] : sum
            ]
    else:
        raise ValueError(
            "xregions (yregions) must have len==2 (len==3) or len==3 (len==2)"
        )
    return A, B, C, D, E, SR


def ABCD_6regions_errorProp(
    abcd, xregions, yregions, sum_var="x", approx=False, new_bins=None
):
    """
    Does 6 region ABCD using error propagation of the statistical uncertanties of the regions.
    """

    A, B, C, D, E, SR = make_ABCD_6regions(abcd, xregions, yregions, sum_var=sum_var)

    # initialize the SR_exp as empty
    SR_exp = SR.copy()
    SR_exp.view().variance = [0] * len(SR.values())
    SR_exp.view().value = [0] * len(SR.values())

    # there are two modes, depending on which dimension is integrated, and which dimension has 3 regions
    # the two modes define different expression for the predicted SR in each bin
    mode1 = (sum_var == "x" and len(xregions) == 3) or (
        sum_var == "y" and len(xregions) == 2
    )
    mode2 = (sum_var == "x" and len(xregions) == 2) or (
        sum_var == "y" and len(xregions) == 3
    )

    # define the histograms that will be used to calculate the scaling factor
    if mode1:
        hNUM = E
        hNUM2 = C
        hDEN = D
    elif mode2:
        hNUM = C
        hNUM2 = E
        hDEN = D
    else:
        raise ValueError(
            "This should not happen. sum_var should be 'x' or 'y', and one of xregions or yregions should have len==2, the other len==3."
        )

    # we need to rebin here in the case appox=True
    if new_bins:
        hNUM = rebin_piecewise(hNUM, new_bins)
        if mode1:
            hDEN = rebin_piecewise(hDEN, new_bins)
        SR = rebin_piecewise(SR, new_bins)
        SR_exp = rebin_piecewise(SR_exp, new_bins)

    preds, preds_err = [], []
    for i in range(len(hNUM.values())):
        hNUM_bin = hNUM[i]
        hDEN_bin = (
            hDEN[i] if mode1 else hist.accumulators.WeightedSum()
        )  # only needed for mode1, if mode 2, just initialize it to an empty accumulator

        # define the scaling factor function
        a, b, hnum_bin, hnum, hnum2, hden_bin, hden = symbols(
            "A B hNUM_bin hNUM hNUM2 hDEN_bin hDEN"
        )
        if mode1 and not approx:
            exp = hnum_bin**2 * hnum2 * a * hden_bin**-1 * b**-2
        elif mode1 and approx:
            exp = hnum_bin * hnum * hnum2 * a * hden**-1 * b**-2
        elif mode2:
            exp = hnum_bin * hnum2**2 * a * b**-2 * hden**-1

        # defines lists of variables (sympy symbols) and accumulators (hist.sum())
        variables = [a, b, hnum_bin, hnum, hnum2, hden_bin, hden]
        accs = [
            A.sum(),
            B.sum(),
            hNUM_bin,
            hNUM.sum(),
            hNUM2.sum(),
            hDEN_bin,
            hDEN.sum(),
        ]

        # calculate scaling factor by substituting values of the histograms' sums for the sympy symbols
        alpha = exp.copy()
        for var, acc in zip(variables, accs):
            alpha = alpha.subs(var, acc.value)

        # calculate the error on the scaling factor
        variance = 0
        for var, acc in zip(variables, accs):
            der = diff(exp, var)
            var = abs(acc.variance)
            variance += der**2 * var
        for var, acc in zip(variables, accs):
            variance = variance.subs(var, acc.value)
        sigma_alpha = variance

        if type(alpha) != sympy.core.numbers.Float or alpha <= 0:
            alpha = 0

        preds.append(alpha)
        preds_err.append(sigma_alpha)

    SR_exp.view().variance = preds_err
    SR_exp.view().value = preds

    return SR, SR_exp


def make_ABCD_9regions(hist_abcd, xregions, yregions, sum_var="X"):
    if sum_var == "x":
        A = hist_abcd[
            xregions[0][0] : xregions[0][1] : sum, yregions[0][0] : yregions[0][1]
        ]
        B = hist_abcd[
            xregions[0][0] : xregions[0][1] : sum, yregions[1][0] : yregions[1][1]
        ]
        C = hist_abcd[
            xregions[0][0] : xregions[0][1] : sum, yregions[2][0] : yregions[2][1]
        ]
        D = hist_abcd[
            xregions[1][0] : xregions[1][1] : sum, yregions[0][0] : yregions[0][1]
        ]
        E = hist_abcd[
            xregions[1][0] : xregions[1][1] : sum, yregions[1][0] : yregions[1][1]
        ]
        F = hist_abcd[
            xregions[1][0] : xregions[1][1] : sum, yregions[2][0] : yregions[2][1]
        ]
        G = hist_abcd[
            xregions[2][0] : xregions[2][1] : sum, yregions[0][0] : yregions[0][1]
        ]
        H = hist_abcd[
            xregions[2][0] : xregions[2][1] : sum, yregions[1][0] : yregions[1][1]
        ]
        SR = hist_abcd[
            xregions[2][0] : xregions[2][1] : sum, yregions[2][0] : yregions[2][1]
        ]

    elif sum_var == "y":
        A = hist_abcd[
            xregions[0][0] : xregions[0][1], yregions[0][0] : yregions[0][1] : sum
        ]
        B = hist_abcd[
            xregions[0][0] : xregions[0][1], yregions[1][0] : yregions[1][1] : sum
        ]
        C = hist_abcd[
            xregions[0][0] : xregions[0][1], yregions[2][0] : yregions[2][1] : sum
        ]
        D = hist_abcd[
            xregions[1][0] : xregions[1][1], yregions[0][0] : yregions[0][1] : sum
        ]
        E = hist_abcd[
            xregions[1][0] : xregions[1][1], yregions[1][0] : yregions[1][1] : sum
        ]
        F = hist_abcd[
            xregions[1][0] : xregions[1][1], yregions[2][0] : yregions[2][1] : sum
        ]
        G = hist_abcd[
            xregions[2][0] : xregions[2][1], yregions[0][0] : yregions[0][1] : sum
        ]
        H = hist_abcd[
            xregions[2][0] : xregions[2][1], yregions[1][0] : yregions[1][1] : sum
        ]
        SR = hist_abcd[
            xregions[2][0] : xregions[2][1], yregions[2][0] : yregions[2][1] : sum
        ]

    return A, B, C, D, E, F, G, H, SR


def ABCD_9regions_errorProp(
    abcd, xregions, yregions, sum_var="x", approx=True, new_bins=None
):
    """
    Does 9 region ABCD using error propagation of the statistical uncertanties of the regions.
    """

    if sum_var == "y":
        raise Exception("sum_var='y' not implemented yet")

    A, B, C, D, E, F, G, H, SR = make_ABCD_9regions(
        abcd, xregions, yregions, sum_var=sum_var
    )
    SR_exp = SR.copy()
    SR_exp.view().variance = [0] * len(SR.values())
    SR_exp.view().value = [0] * len(SR.values())

    # we need to rebin here in the case appox=True
    if new_bins:
        if sum_var == "x":
            F = rebin_piecewise(F, new_bins)
            C = rebin_piecewise(C, new_bins)
            SR = rebin_piecewise(SR, new_bins)
            SR_exp = rebin_piecewise(SR_exp, new_bins)

    preds, preds_err = [], []
    for i in range(len(F.values())):
        # this is needed in order to do error propagation correctly
        F_bin = F[i]
        C_bin = C[i]
        F_other = F.copy()
        F_other[i] = hist.accumulators.WeightedSum()

        # define the scaling factor function
        a, b, c, c_bin, d, e, f_bin, f_other, g, h = symbols(
            "A B C C_bin D E F_bin F_other G H"
        )
        if sum_var == "x" and approx:
            exp = (
                f_bin
                * (f_other + f_bin)
                * h**2
                * d**2
                * b**2
                * g**-1
                * c**-1
                * a**-1
                * e**-4
            )
        elif sum_var == "x" and not approx:
            exp = f_bin**2 * h**2 * d**2 * b**2 * g**-1 * c_bin**-1 * a**-1 * e**-4
        elif sum_var == "y":
            pass

        # defines lists of variables (sympy symbols) and accumulators (hist.sum())
        variables = [a, b, c, c_bin, d, e, f_bin, f_other, g, h]
        accs = [
            A.sum(),
            B.sum(),
            C.sum(),
            C_bin,
            D.sum(),
            E.sum(),
            F_bin,
            F_other.sum(),
            G.sum(),
            H.sum(),
        ]

        # calculate scaling factor by substituting values of the histograms' sums for the sympy symbols
        alpha = exp.copy()
        for var, acc in zip(variables, accs):
            alpha = alpha.subs(var, acc.value)

        # calculate the error on the scaling factor
        variance = 0
        for var, acc in zip(variables, accs):
            der = diff(exp, var)
            var = abs(acc.variance)
            variance += der**2 * var
        for var, acc in zip(variables, accs):
            variance = variance.subs(var, acc.value)
        sigma_alpha = variance

        if type(alpha) != sympy.core.numbers.Float or alpha <= 0:
            alpha = 0

        preds.append(alpha)
        preds_err.append(sigma_alpha)

    SR_exp.view().variance = preds_err
    SR_exp.view().value = preds

    return SR, SR_exp
