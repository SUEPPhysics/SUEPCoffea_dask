from coffea.lookup_tools.dense_lookup import dense_lookup


def make_btag_effs(
    h, hadron_flavors={0: "L", 4: "C", 5: "B"}, btag_categories={1: "L", 2: "T"}
):
    """
    Take as input a 4D histogram with axes jets_pt, jets_eta, jets_hadronFlavor (default: 0: loose, 1: charm, 4: bottom), jets_btag_category (default: 0: fail, 1: loose, 2: tight).
    The assumption is that the btag categories are in increasing order of tightness, such that efficiency in category i is the number of jets in i divided by any bin < i.
    Calculate b tagging efficiencies for each btag category, for each hadron flavor, as functions of pt and eta.
    Returns a dictionary of coffea dense_lookup correctors.
    """

    effs = {}
    for btag, blabel in btag_categories.items():
        effs[blabel] = {}
        for flavor, flabel in hadron_flavors.items():
            num = h[:, :, flavor * 1.0j, btag * 1.0j].values()
            den = h[:, :, flavor * 1.0j, : btag * 1.0j : sum].values()
            eff = np.where(den > 0, num / den, 0)
            corr = dense_lookup(eff, [h.axes[0].edges, h.axes[1].edges])
            effs[blabel][flabel] = corr

    return effs
