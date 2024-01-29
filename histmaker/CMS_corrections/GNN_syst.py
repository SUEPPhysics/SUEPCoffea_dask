import json
import logging
import sys

import hist

sys.path.append("../../")
import plotting.plot_utils


def apply_GNN_syst(plots, fGNNsyst, models, bins, era, out_label="GNN"):
    """
    Applies GNN scaling from file to every histogram whose name ends with "GNN",
    making both down and up variations.

    Inputs:
        plots: dictionary of Hist histograms
        fGNNsyst: str, path to the file containing the systematic
        models: list of str, list of model names to find in the fGNNsyst
        bins: list of complex numbers, to use as slicing in the histogram
        era: int, year
        out_label: str, output label of method to apply systematic to

    Outputs:
        Updated plots dictionary
    """

    # load in the json file containing the corrections for each year/model
    with open(fGNNsyst) as f:
        GNNsyst = json.load(f)

    # complex numbers for hist
    for model in models:
        # load the correct model for each year
        yearSyst = GNNsyst.get(str(era))
        if yearSyst is None:
            logging.warning(
                "--- {} was not found in file {}; systematic has not been applied".format(
                    era, fGNNsyst
                )
            )
            continue
        scales = yearSyst.get(model)
        if scales is None:
            logging.warning(
                "--- {} was not found in file {}; systematic has not been applied".format(
                    model, fGNNsyst
                )
            )
            continue

        # scale them
        GNN_syst_plots = {}
        for plot in plots.keys():
            # apply only to GNN
            if not plot.endswith(out_label):
                continue

            if model in plot and "2D" not in plot:
                GNN_syst_plots[plot + "_GNNsyst_down"] = (
                    plot_utils.apply_binwise_scaling(
                        plots[plot].copy(), bins, [1 - s for s in scales]
                    )
                )
                GNN_syst_plots[plot + "_GNNsyst_up"] = plot_utils.apply_binwise_scaling(
                    plots[plot].copy(), bins, [1 + s for s in scales]
                )
            if model in plot and "2D" in plot:
                var1 = plot.split("_vs_")[0]
                var2 = plot.split("_vs_")[1]
                if model in var1:
                    dim = "x"
                elif model in var2:
                    dim = "y"
                GNN_syst_plots[plot + "_GNNsyst_down"] = (
                    plot_utils.apply_binwise_scaling(
                        plots[plot].copy(), bins, [1 - s for s in scales], dim=dim
                    )
                )
                GNN_syst_plots[plot + "_GNNsyst_up"] = plot_utils.apply_binwise_scaling(
                    plots[plot].copy(), bins, [1 + s for s in scales], dim=dim
                )
        plots.update(GNN_syst_plots)

    return plots
