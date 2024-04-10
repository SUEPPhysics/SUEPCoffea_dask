from hist import Hist


def initialize_histograms(output: dict, label: str, options, config: dict) -> dict:
    # don't recreate histograms if called multiple times with the same output label
    if label in output["labels"]:
        return output
    else:
        output["labels"].append(label)

    regions_list = [""]
    if options.doABCD:
        init_hists_ABCD(output, label, config)
        regions_list += get_ABCD_regions(config)

    ###################################################################3#######################################################
    init_hists_default(output, label, regions_list)

    ###########################################################################################################################
    if "ClusterInverted" in label:
        init_hists_clusterInverted(output, label, regions_list)

    ###########################################################################################################################
    if label == "GNN" and options.doInf:
        init_hists_GNN(output, label, config, regions_list)

    ###########################################################################################################################
    if label == "GNNInverted" and options.doInf:
        init_hists_GNNInverted(output, label, config, regions_list)

    ###########################################################################################################################
    if options.channel == "WH":
        init_hists_highestPT(output, label)

    return output


def init_hists_ABCD(output: dict, label: str, config: dict):
    xvar = config["xvar"]
    yvar = config["yvar"]
    xvar_regions = config["xvar_regions"]
    yvar_regions = config["yvar_regions"]
    output.update(
        {
            f"2D_{xvar}_vs_{yvar}_{label}": Hist.new.Reg(
                100, xvar_regions[0], xvar_regions[-1], name=xvar
            )
            .Reg(100, yvar_regions[0], yvar_regions[-1], name=yvar)
            .Weight()
        }
    )


def get_ABCD_regions(config: dict) -> list:
    xvar_regions = config["xvar_regions"]
    yvar_regions = config["yvar_regions"]
    regions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n_regions = (len(xvar_regions) - 1) * (len(yvar_regions) - 1)
    return [regions[i] + "_" for i in range(n_regions)]


def init_hists_default(output: dict, label: str, regions_list: list = [""]) -> None:
    """
    Initialize histograms for all the variables we want to plot.
    Parameters:
        output (dict): dictionary of histograms
        label (str): label for the histograms. This is usually some label name + systematic name
        regions_list (list): list of regions to make histograms for. This is usually just [""], but can be ABCD regions ["A", "B", ...]
    """

    for r in regions_list:
        # these histograms will be made for each systematic, and each ABCD region
        output.update(
            {
                f"{r}SUEP_nconst_{label}": Hist.new.Reg(
                    501,
                    0,
                    500,
                    name=f"{r}SUEP_nconst_{label}",
                    label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
                ).Weight(),
                f"{r}SUEP_S1_{label}": Hist.new.Reg(
                    100,
                    0,
                    1,
                    name=f"{r}SUEP_S1_{label}",
                    label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
                ).Weight(),
            }
        )

        # these histograms will be made for each systematic, and no ABCD region
        if r == "":
            output.update(
                {
                    f"ht_{label}": Hist.new.Reg(
                        100, 0, 10000, name=f"ht_{label}", label="HT"
                    ).Weight(),
                    f"ht_JEC_{label}": Hist.new.Reg(
                        100, 0, 10000, name=f"ht_JEC_{label}", label="HT JEC"
                    ).Weight(),
                    f"ht_JEC_JER_up_{label}": Hist.new.Reg(
                        100,
                        0,
                        10000,
                        name=f"ht_JEC_JER_up_{label}",
                        label="HT JEC up",
                    ).Weight(),
                    f"ht_JEC_JER_down_{label}": Hist.new.Reg(
                        100,
                        0,
                        10000,
                        name=f"ht_JEC_JER_down_{label}",
                        label="HT JEC JER down",
                    ).Weight(),
                    f"ht_JEC_JES_up_{label}": Hist.new.Reg(
                        100,
                        0,
                        10000,
                        name=f"ht_JEC_JES_up_{label}",
                        label="HT JEC JES up",
                    ).Weight(),
                    f"ht_JEC_JES_down_{label}": Hist.new.Reg(
                        100,
                        0,
                        10000,
                        name=f"ht_JEC_JES_down_{label}",
                        label="HT JEC JES down",
                    ).Weight(),
                    f"ntracks_{label}": Hist.new.Reg(
                        101,
                        0,
                        500,
                        name=f"ntracks_{label}",
                        label="# Tracks in Event",
                    ).Weight(),
                    f"ngood_fastjets_{label}": Hist.new.Reg(
                        10,
                        0,
                        10,
                        name=f"ngood_fastjets_{label}",
                        label="# AK15 jets in Event",
                    ).Weight(),
                    f"PV_npvs_{label}": Hist.new.Reg(
                        200,
                        0,
                        200,
                        name=f"PV_npvs_{label}",
                        label="# PVs in Event ",
                    ).Weight(),
                    f"Pileup_nTrueInt_{label}": Hist.new.Reg(
                        200,
                        0,
                        200,
                        name=f"Pileup_nTrueInt_{label}",
                        label="# True Interactions in Event ",
                    ).Weight(),
                    f"ngood_ak4jets_{label}": Hist.new.Reg(
                        20,
                        0,
                        20,
                        name=f"ngood_ak4jets_{label}",
                        label="# ak4jets in Event",
                    ).Weight(),
                    f"ngood_ak4jets_noLepIso_{label}": Hist.new.Reg(
                        20,
                        0,
                        20,
                        name=f"ngood_ak4jets_noLepIso_{label}",
                        label="# ak4jets in Event (no lepton iso.)",
                    ).Weight(),
                    f"2D_SUEP_S1_vs_SUEP_nconst_{label}": Hist.new.Reg(
                        100,
                        0,
                        1.0,
                        name=f"SUEP_S1_{label}",
                        label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
                    )
                    .Reg(
                        501,
                        0,
                        500,
                        name=f"nconst_{label}",
                        label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
                    )
                    .Weight(),
                    f"2D_SUEP_S1_vs_ntracks_{label}": Hist.new.Reg(
                        100,
                        0,
                        1.0,
                        name=f"SUEP_S1_{label}",
                        label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
                    )
                    .Reg(100, 0, 500, name=f"ntracks_{label}", label="# Tracks")
                    .Weight(),
                    f"2D_PV_npvs_vs_SUEP_nconst_{label}": Hist.new.Reg(
                        200, 0, 200, name=f"PV_npvs_{label}", label="# PVs in Event"
                    )
                    .Reg(
                        501,
                        0,
                        500,
                        name=f"nconst_{label}",
                        label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
                    )
                    .Weight(),
                    f"2D_PV_npvs_vs_ntracks_{label}": Hist.new.Reg(
                        200, 0, 200, name=f"PV_npvs_{label}", label="# PVs in Event"
                    )
                    .Reg(100, 0, 500, name=f"ntracks_{label}", label="# Tracks")
                    .Weight(),
                }
            )

        # these histograms will be made for only nominal, and no ABCD regions
        if (
            r == ""
            and not label.lower().endswith("up")
            and not label.lower().endswith("down")
        ):
            output.update(
                {
                    f"{r}SUEP_genMass_{label}": Hist.new.Reg(
                        100,
                        0,
                        1200,
                        name=f"{r}SUEP_genMass_{label}",
                        label="Gen Mass of SUEP ($m_S$) [GeV]",
                    ).Weight(),
                    f"{r}SUEP_pt_{label}": Hist.new.Reg(
                        1000,
                        0,
                        1000,
                        name=f"{r}SUEP_pt_{label}",
                        label=r"SUEP $p_T$ [GeV]",
                    ).Weight(),
                    f"{r}SUEP_delta_pt_genPt_{label}": Hist.new.Reg(
                        400,
                        -2000,
                        2000,
                        name=f"{r}SUEP_delta_pt_genPt_{label}",
                        label="SUEP $p_T$ - genSUEP $p_T$ [GeV]",
                    ).Weight(),
                    f"{r}SUEP_pt_avg_{label}": Hist.new.Reg(
                        500,
                        0,
                        100,
                        name=f"{r}SUEP_pt_avg_{label}",
                        label="SUEP Components $p_T$ Avg.",
                    ).Weight(),
                    f"{r}SUEP_pt_avg_b_{label}": Hist.new.Reg(
                        100,
                        0,
                        20,
                        name=f"{r}SUEP_pt_avg_b_{label}",
                        label=r"SUEP Components $p_T^{\mathrm{boosted}}$ Avg.",
                    ).Weight(),
                    f"{r}SUEP_eta_{label}": Hist.new.Reg(
                        100,
                        -5,
                        5,
                        name=f"{r}SUEP_eta_{label}",
                        label=r"SUEP $\eta$",
                    ).Weight(),
                    f"{r}SUEP_phi_{label}": Hist.new.Reg(
                        100,
                        -6.5,
                        6.5,
                        name=f"{r}SUEP_phi_{label}",
                        label=r"SUEP $\phi$",
                    ).Weight(),
                    f"{r}SUEP_mass_{label}": Hist.new.Reg(
                        150,
                        0,
                        2000,
                        name=f"{r}SUEP_mass_{label}",
                        label="SUEP Mass [GeV]",
                    ).Weight(),
                    f"{r}SUEP_delta_mass_genMass_{label}": Hist.new.Reg(
                        400,
                        -2000,
                        2000,
                        name=f"{r}SUEP_delta_mass_genMass_{label}",
                        label="SUEP Mass - genSUEP Mass [GeV]",
                    ).Weight(),
                    f"{r}SUEP_ISR_deltaPhi_{label}": Hist.new.Reg(
                        400,
                        -6,
                        6,
                        name=f"{r}SUEP_ISR_deltaPhi_{label}",
                        label=r"SUEP $\phi$ - ISR $\phi$ [GeV]",
                    ).Weight(),
                }
            )


def init_hists_clusterInverted(output, label, regions_list=[""]):
    output.update(
        {
            # 2D histograms
            f"2D_ISR_S1_vs_ntracks_{label}": Hist.new.Reg(
                100,
                0,
                1.0,
                name=f"ISR_S1_{label}",
                label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
            )
            .Reg(200, 0, 500, name=f"ntracks_{label}", label="# Tracks")
            .Weight(),
            f"2D_ISR_S1_vs_ISR_nconst_{label}": Hist.new.Reg(
                100,
                0,
                1.0,
                name=f"ISR_S1_{label}",
                label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
            )
            .Reg(
                501,
                0,
                500,
                name=f"nconst_{label}",
                label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
            )
            .Weight(),
            f"2D_ISR_nconst_vs_ISR_pt_avg_{label}": Hist.new.Reg(
                501, 0, 500, name=f"ISR_nconst_{label}"
            )
            .Reg(500, 0, 500, name=f"ISR_pt_avg_{label}")
            .Weight(),
        }
    )
    # variables from the dataframe for all the events, and those in A, B, C regions
    for r in regions_list:
        output.update(
            {
                f"{r}ISR_nconst_{label}": Hist.new.Reg(
                    501,
                    0,
                    500,
                    name=f"{r}ISR_nconst_{label}",
                    label="# Tracks in ISR",
                ).Weight(),
                f"{r}ISR_S1_{label}": Hist.new.Reg(
                    100,
                    0,
                    1,
                    name=f"{r}ISR_S1_{label}",
                    label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
                ).Weight(),
            }
        )
        if (
            "up" not in label and "down" not in label and r == ""
        ):  # don't care for systematics for these
            output.update(
                {
                    f"ISR_pt_{label}": Hist.new.Reg(
                        100,
                        0,
                        2000,
                        name=f"ISR_pt_{label}",
                        label=r"ISR $p_T$ [GeV]",
                    ).Weight(),
                    f"ISR_pt_avg_{label}": Hist.new.Reg(
                        500,
                        0,
                        500,
                        name=f"ISR_pt_avg_{label}",
                        label=r"ISR Components $p_T$ Avg.",
                    ).Weight(),
                    f"ISR_eta_{label}": Hist.new.Reg(
                        100,
                        -5,
                        5,
                        name=f"ISR_eta_{label}",
                        label=r"ISR $\eta$",
                    ).Weight(),
                    f"ISR_phi_{label}": Hist.new.Reg(
                        100,
                        -6.5,
                        6.5,
                        name=f"ISR_phi_{label}",
                        label=r"ISR $\phi$",
                    ).Weight(),
                    f"ISR_mass_{label}": Hist.new.Reg(
                        150,
                        0,
                        4000,
                        name=f"ISR_mass_{label}",
                        label="ISR Mass [GeV]",
                    ).Weight(),
                }
            )


def init_hists_GNN(output, label, config, regions_list=[""]):
    # 2D histograms
    for model in config["models"]:
        output.update(
            {
                f"2D_SUEP_S1_vs_{model}_{label}": Hist.new.Reg(
                    100,
                    0,
                    1.0,
                    name=f"SUEP_S1_{label}",
                    label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
                )
                .Reg(100, 0, 1, name=f"{model}_{label}", label="GNN Output")
                .Weight(),
                f"2D_SUEP_nconst_vs_{model}_{label}": Hist.new.Reg(
                    501,
                    0,
                    500,
                    name=f"SUEP_nconst_{label}",
                    label="# Const",
                )
                .Reg(100, 0, 1, name=f"{model}_{label}", label="GNN Output")
                .Weight(),
            }
        )

    output.update(
        {
            f"2D_SUEP_nconst_vs_SUEP_S1_{label}": Hist.new.Reg(
                501, 0, 500, name=f"SUEP_nconst_{label}", label="# Const"
            )
            .Reg(
                100,
                0,
                1,
                name=f"SUEP_S1_{label}",
                label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
            )
            .Weight(),
        }
    )

    for r in regions_list:
        output.update(
            {
                f"{r}SUEP_nconst_{label}": Hist.new.Reg(
                    501,
                    0,
                    500,
                    name=f"{r}SUEP_nconst{label}",
                    label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
                ).Weight(),
                f"{r}SUEP_S1_{label}": Hist.new.Reg(
                    100,
                    -1,
                    2,
                    name=f"{r}SUEP_S1_{label}",
                    label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
                ).Weight(),
            }
        )
        for model in config["models"]:
            output.update(
                {
                    f"{r}{model}_{label}": Hist.new.Reg(
                        100,
                        0,
                        1,
                        name=f"{r}{model}_{label}",
                        label="GNN Output",
                    ).Weight()
                }
            )


def init_hists_GNNInverted(output, label, config, regions_list=[""]):
    # 2D histograms
    for model in config["models"]:
        output.update(
            {
                f"2D_ISR_S1_vs_{model}_{label}": Hist.new.Reg(
                    100,
                    0,
                    1.0,
                    name=f"ISR_S1_{label}",
                    label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
                )
                .Reg(100, 0, 1, name=f"{model}_{label}", label="GNN Output")
                .Weight(),
                f"2D_ISR_nconst_vs_{model}_{label}": Hist.new.Reg(
                    501, 0, 500, name=f"ISR_nconst_{label}", label="# Const"
                )
                .Reg(100, 0, 1, name=f"{model}_{label}", label="GNN Output")
                .Weight(),
            }
        )
    output.update(
        {
            f"2D_ISR_nconst_vs_ISR_S1_{label}": Hist.new.Reg(
                501, 0, 500, name=f"ISR_nconst_{label}", label="# Const"
            )
            .Reg(
                100,
                0,
                1,
                name=f"ISR_S1_{label}",
                label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
            )
            .Weight()
        }
    )

    for r in regions_list:
        output.update(
            {
                f"{r}ISR_nconst_{label}": Hist.new.Reg(
                    501,
                    0,
                    500,
                    name=f"{r}ISR_nconst{label}",
                    label="# Tracks in ISR",
                ).Weight(),
                f"{r}ISR_S1_{label}": Hist.new.Reg(
                    100,
                    -1,
                    2,
                    name=f"{r}ISR_S1_{label}",
                    label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
                ).Weight(),
            }
        )
        for model in config["models"]:
            output.update(
                {
                    f"{r}{model}_{label}": Hist.new.Reg(
                        100,
                        0,
                        1,
                        name=f"{r}{model}_{label}",
                        label="GNN Output",
                    ).Weight()
                }
            )


def init_hists_highestPT(output, label, regions_list=[""]):
    output.update(
        {
            f"SUEP_nconst_{label}": Hist.new.Reg(
                100,
                0,
                100,
                name=f"{r}SUEP_nconst_{label}",
                label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
            ).Weight(),
            f"CaloMET_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"CaloMET_pt_{label}",
                label="CaloMET $p_T$",
            ).Weight(),
            f"CaloMET_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"CaloMET_phi_{label}",
                label=r"CaloMET $\phi$",
            ).Weight(),
            f"CaloMET_sumEt_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"CaloMET_sumEt_{label}",
                label="CaloMET sumEt",
            ).Weight(),
            f"PuppiMET_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"PuppiMET_pt_{label}",
                label="PuppiMET $p_T$",
            ).Weight(),
            #            '''f"PuppiMET_pt_JER_up_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"PuppiMET_pt_JER_up_{label}",
            #                label="PuppiMET $p_T$",
            #            ).Weight(),
            #            f"PuppiMET_pt_JER_down_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"PuppiMET_pt_JER_down_{label}",
            #                label="PuppiMET $p_T$",
            #            ).Weight(),
            #            f"PuppiMET_pt_JES_up_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"PuppiMET_pt_JES_up{label}",
            #                label="PuppiMET $p_T$",
            #            ).Weight(),
            #            f"PuppiMET_pt_JES_down_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"PuppiMET_pt_JES_down{label}",
            #                label="PuppiMET $p_T$",
            #            ).Weight(),'''
            f"PuppiMET_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"PuppiMET_phi_{label}",
                label=r"PuppiMET $\phi$",
            ).Weight(),
            #            f"PuppiMET_phi_JER_up_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"PuppiMET_phi_JER_up_{label}",
            #                label="PuppiMET $\phi$",
            #            ).Weight(),
            #            f"PuppiMET_phi_JER_down_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"PuppiMET_phi_JER_down_{label}",
            #                label="PuppiMET $\phi$",
            #            ).Weight(),
            #            f"PuppiMET_phi_JES_up_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"PuppiMET_phi_JES_up_{label}",
            #                label="PuppiMET $\phi$",
            #            ).Weight(),
            #            f"PuppiMET_phi_JES_down_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"PuppiMET_phi_JES_down_{label}",
            #                label="PuppiMET $\phi$",
            #            ).Weight(),'''
            f"PuppiMET_sumEt_{label}": Hist.new.Reg(
                100,
                0,
                5000,
                name=f"PuppiMET_sumEt_{label}",
                label="PuppiMET sumEt",
            ).Weight(),
            f"MET_pt_{label}": Hist.new.Reg(
                1000,
                0,
                1000,
                name=f"MET_pt_{label}",
                label="MET $p_T$",
            ).Weight(),
            f"MET_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"MET_phi_{label}",
                label=r"MET $\phi$",
            ).Weight(),
            f"nBLoose_{label}": Hist.new.Reg(
                20,
                0,
                20,
                name=f"nBLoose_{label}",
                label="# loose cut b jets",
            ).Weight(),
            f"nBMedium_{label}": Hist.new.Reg(
                20,
                0,
                20,
                name=f"nBMedium_{label}",
                label="# medium cut b jets",
            ).Weight(),
            f"nBTight_{label}": Hist.new.Reg(
                20,
                0,
                20,
                name=f"nBTight_{label}",
                label="# tight cut b jets",
            ).Weight(),
            f"nBLoose_noLepIso_{label}": Hist.new.Reg(
                20,
                0,
                20,
                name=f"nBLoose_noLepIso_{label}",
                label="# loose cut b jets",
            ).Weight(),
            f"nBMedium_noLepIso_{label}": Hist.new.Reg(
                20,
                0,
                20,
                name=f"nBMedium_noLepIso_{label}",
                label="# medium cut b jets",
            ).Weight(),
            f"nBTight_noLepIso_{label}": Hist.new.Reg(
                20,
                0,
                20,
                name=f"nBTight_noLepIso_{label}",
                label="# tight cut b jets",
            ).Weight(),
            f"jet1_pt_{label}": Hist.new.Reg(
                1000,
                0,
                2000,
                name=f"jet1_pt_{label}",
                label="leading jet $p_T$",
            ).Weight(),
            f"jet1_phi_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"jet1_phi_{label}",
                label=r"leading jet $\phi$",
            ).Weight(),
            f"jet1_eta_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"jet1_eta_{label}",
                label=r"leading jet $\eta$",
            ).Weight(),
            f"jet1_qgl_{label}": Hist.new.Reg(
                20,
                0,
                1,
                name=f"jet1_qgl_{label}",
                label="leading jet qgl",
            ).Weight(),
            f"jet1_mass_{label}": Hist.new.Reg(
                1000,
                0,
                1000,
                name=f"jet1_mass_{label}",
                label="leading jet mass",
            ).Weight(),
            f"jet2_pt_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"jet2_pt_{label}",
                label="2nd leading jet $p_T$",
            ).Weight(),
            f"jet2_phi_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"jet2_phi_{label}",
                label=r"2nd leading jet $\phi$",
            ).Weight(),
            f"jet2_eta_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"jet2_eta_{label}",
                label=r"2nd leading jet $\eta$",
            ).Weight(),
            f"jet2_qgl_{label}": Hist.new.Reg(
                20,
                0,
                1,
                name=f"jet2_qgl_{label}",
                label="2nd leading jet qgl",
            ).Weight(),
            f"jet3_pt_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"jet3_pt_{label}",
                label="3rd leading jet $p_T$",
            ).Weight(),
            f"jet3_phi_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"jet3_phi_{label}",
                label=r"3rd leading jet $\phi$",
            ).Weight(),
            f"jet3_eta_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"jet3_eta_{label}",
                label=r"3rd leading jet $\eta$",
            ).Weight(),
            f"jet3_qgl_{label}": Hist.new.Reg(
                20,
                0,
                1,
                name=f"jet3_qgl_{label}",
                label="3rd leading jet qgl",
            ).Weight(),
            f"bjet_pt_{label}": Hist.new.Reg(
                1000,
                0,
                2000,
                name=f"bjet_pt_{label}",
                label="highest btag jet $p_T$",
            ).Weight(),
            f"bjet_phi_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"bjet_phi_{label}",
                label=r"highest btag jet $\phi$",
            ).Weight(),
            f"bjet_eta_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"bjet_eta_{label}",
                label=r"highest btag jet $\eta$",
            ).Weight(),
            f"bjet_qgl_{label}": Hist.new.Reg(
                20,
                0,
                1,
                name=f"bjet_qgl_{label}",
                label="highest btag jet qgl",
            ).Weight(),
            f"bjet_btag_{label}": Hist.new.Reg(
                20,
                0,
                1,
                name=f"bjet_btag_{label}",
                label="highest btag jet btag value",
            ).Weight(),
            f"ak4jets_inSUEPcluster_n_{label}": Hist.new.Reg(
                20,
                0,
                20,
                name=f"ak4jets_inSUEPcluster_n_{label}",
                label="# ak4jets in SUEP cluster",
            ).Weight(),
            f"ak4jets_inSUEPcluster_pt_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"ak4jets_inSUEPcluster_pt_{label}",
                label="ak4jets in SUEP cluster $p_T$",
            ).Weight(),
            f"ak4jet1_inSUEPcluster_pt_{label}": Hist.new.Reg(
                1000,
                0,
                1000,
                name=f"ak4jet1_inSUEPcluster_pt_{label}",
                label="ak4jet1 in SUEP cluster $p_T$",
            ).Weight(),
            f"ak4jet1_inSUEPcluster_mass_{label}": Hist.new.Reg(
                1000,
                0,
                1000,
                name=f"ak4jet1_inSUEPcluster_mass_{label}",
                label="ak4jet1 in SUEP cluster mass",
            ).Weight(),
            f"ak4jet2_inSUEPcluster_pt_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"ak4jet2_inSUEPcluster_pt_{label}",
                label="ak4jet2 in SUEP cluster $p_T$",
            ).Weight(),
            f"nak4jets_outsideSUEP_{label}": Hist.new.Reg(
                20,
                0,
                20,
                name=f"nak4jets_outsideSUEP_n_{label}",
                label="# AK4 jets outside SUEP AK15 cluster",
            ).Weight(),
            f"minDeltaPhiMETJet_pt_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"minDeltaPhiMETJet_pt_{label}",
                label=r"$\mathrm{jet}^{\mathrm{closest~to~MET}}$ $p_T$",
            ).Weight(),
            f"minDeltaPhiMETJet_phi_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"minDeltaPhiMETJet_phi_{label}",
                label=r"$\mathrm{jet}^{\mathrm{closest~to~MET}}$ $\phi$",
            ).Weight(),
            f"minDeltaPhiMETJet_qgl_{label}": Hist.new.Reg(
                20,
                0,
                1,
                name=f"minDeltaPhiMETJet_qgl_{label}",
                label=r"$\mathrm{jet}^{\mathrm{closest~to~MET}}$ qgl",
            ).Weight(),
            f"nLooseLeptons_{label}": Hist.new.Reg(
                10,
                0,
                10,
                name=f"nLooseLeptons_{label}",
                label="# loose leptons",
            ).Weight(),
            f"lepton_pt_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"lepton_pt_{label}",
                label=r"$p_T^{\ell}$",
            ).Weight(),
            f"lepton_phi_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"lepton_phi_{label}",
                label=r"$\phi^{\ell}$",
            ).Weight(),
            f"lepton_eta_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"lepton_eta_{label}",
                label=r"$\eta^{\ell}$",
            ).Weight(),
            f"lepton_mass_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"lepton_mass_{label}",
                label=r"$m^{\ell}$",
            ).Weight(),
            f"lepton_flavor_{label}": Hist.new.Reg(
                40,
                -20,
                20,
                name=f"lepton_flavor_{label}",
                label=r"$\ell$ pdgID",
            ).Weight(),
            f"lepton_ID_{label}": Hist.new.Reg(
                10,
                0,
                10,
                name=f"lepton_ID_{label}",
                label=r"$\ell$ ID",
            ).Weight(),
            f"lepton_IDMVA_{label}": Hist.new.Reg(
                10,
                0,
                10,
                name=f"lepton_IDMVA_{label}",
                label=r"$\ell$ IDMVA",
            ).Weight(),
            f"lepton_iso_{label}": Hist.new.Reg(
                50,
                0,
                1,
                name=f"lepton_iso_{label}",
                label=r"$\ell$ iso",
            ).Weight(),
            f"lepton_miniIso_{label}": Hist.new.Reg(
                50,
                0,
                1,
                name=f"lepton_iso_{label}",
                label=r"$\ell$ miniIso",
            ).Weight(),
            f"lepton_isoMVA_{label}": Hist.new.Reg(
                50,
                0,
                1,
                name=f"lepton_isoMVA_{label}",
                label=r"$\ell$ isoMVA",
            ).Weight(),
            f"lepton_dxy_{label}": Hist.new.Reg(
                100,
                -0.2,
                0.2,
                name=f"lepton_dxy_{label}",
                label=r"$\ell$ dxy",
            ).Weight(),
            f"lepton_dz_{label}": Hist.new.Reg(
                100,
                -0.2,
                0.2,
                name=f"lepton_dz_{label}",
                label=r"$\ell$ dz",
            ).Weight(),
            f"nphotons_{label}": Hist.new.Int(
                0,
                10,
                name=f"nphotons_{label}",
                label=r"# $\gamma$",
            ).Weight(),
            f"photon1_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"photon1_pt_{label}",
                label=r"$\gamma^{\mathrm{leading}}$ $p_T$",
            ).Weight(),
            f"photon1_eta_{label}": Hist.new.Reg(
                100,
                -3,
                3,
                name=f"photon1_eta_{label}",
                label=r"$\gamma^{\mathrm{leading}}$ $\eta$",
            ).Weight(),
            f"photon1_pixelSeed_{label}": Hist.new.Reg(
                2,
                0,
                2,
                name=f"photon1_pixelSeed_{label}",
                label=r"$\gamma^{\mathrm{leading}}$ pixelSeed",
            ).Weight(),
            f"photon1_mvaID_{label}": Hist.new.Reg(
                100,
                -1,
                1,
                name=f"photon1_mvaID_{label}",
                label=r"$\gamma^{\mathrm{leading}}$ mvaID",
            ).Weight(),
            f"photon1_electronVeto_{label}": Hist.new.Reg(
                2,
                0,
                2,
                name=f"photon1_electronVeto_{label}",
                label=r"$\gamma^{\mathrm{leading}}$ electronVeto",
            ).Weight(),
            f"photon1_hoe_{label}": Hist.new.Reg(
                100,
                0,
                1,
                name=f"photon1_hoe_{label}",
                label=r"$\gamma^{\mathrm{leading}}$ HoE",
            ).Weight(),
            f"photon1_r9_{label}": Hist.new.Reg(
                100,
                0,
                1.5,
                name=f"photon1_r9_{label}",
                label=r"$\gamma^{\mathrm{leading}}$ r9",
            ).Weight(),
            f"photon1_cutBased_{label}": Hist.new.Reg(
                4,
                0,
                4,
                name=f"photon1_cutBased_{label}",
                label=r"$\gamma^{\mathrm{leading}}$ cutBased",
            ).Weight(),
            f"photon1_pfRelIso03_all_{label}": Hist.new.Reg(
                100,
                0,
                5,
                name=f"photon1_pfRelIso03_all_{label}",
                label=r"$\gamma^{\mathrm{leading}}$ pfRelIso03_all",
            ).Weight(),
            f"photon1_isScEtaEB_{label}": Hist.new.Reg(
                2,
                0,
                2,
                name=f"photon1_isScEtaEB_{label}",
                label=r"$\gamma^{\mathrm{leading}}$ isScEtaEB",
            ).Weight(),
            f"photon1_isScEtaEE_{label}": Hist.new.Reg(
                2,
                0,
                2,
                name=f"photon1_isScEtaEE_{label}",
                label=r"$\gamma^{\mathrm{leading}}$ isScEtaEE",
            ).Weight(),
            f"minDeltaR_ak4jet_photon1_{label}": Hist.new.Reg(
                100,
                0,
                6,
                name=f"minDeltaR_ak4jet_photon1_{label}",
                label=r"min($\Delta R$(ak4jets, $\gamma$))",
            ).Weight(),
            f"minDeltaR_lepton_photon1_{label}": Hist.new.Reg(
                100,
                0,
                6,
                name=f"minDeltaR_lepton_photon1_{label}",
                label=r"min($\Delta R$($\ell$, $\gamma$))",
            ).Weight(),
            f"W_SUEP_BV_{label}": Hist.new.Reg(
                100,
                -1,
                10,
                name=f"W_SUEP_BV_{label}",
                label="($p_T^W - p_T^{SUEP}$)/$p_T^{SUEP}$",
            ).Weight(),
            f"W_jet1_BV_{label}": Hist.new.Reg(
                100,
                -1,
                10,
                name=f"W_jet1_BV_{label}",
                label="($p_T^W - p_T^{jet1}$)/$p_T^{jet1}$",
            ).Weight(),
            f"ak4SUEP1_SUEP_BV_{label}": Hist.new.Reg(
                100,
                -1,
                5,
                name=f"SUEP_ak4SUEP1_BV_{label}",
                label=r"($p_T^{\mathrm{leading~ak4~jet~in~SUEP}} - p_T^{SUEP}$)/$p_T^{SUEP}$",
            ).Weight(),
            f"W_SUEP_vBV_{label}": Hist.new.Reg(
                100,
                -1,
                10,
                name=f"W_SUEP_vBV_{label}",
                label="($\\vec{p}_T^W - \\vec{p}_T^{SUEP}$)/$|\\vec{p}_T^{SUEP}|$",
            ).Weight(),
            f"W_jet1_vBV_{label}": Hist.new.Reg(
                100,
                -1,
                10,
                name=f"W_jet1_vBV_{label}",
                label="($\\vec{p}_T^W - \\vec{p}_T^{jet1}$)/$|\\vec{p}_T^{jet1}|$",
            ).Weight(),
            f"W_mT_from_CaloMET_{label}": Hist.new.Reg(
                201,
                0,
                200,
                name=f"W_mT_from_CaloMET_{label}",
                label=r" $W^{\mathrm{CaloMET}} m_T$",
            ).Weight(),
            f"W_mT_from_PuppiMET_{label}": Hist.new.Reg(
                201,
                0,
                200,
                name=f"W_mT_from_PuppiMET_{label}",
                label=r" $W^{\mathrm{PuppiMET}} m_T$",
            ).Weight(),
            f"W_mT_from_MET_{label}": Hist.new.Reg(
                201,
                0,
                200,
                name=f"W_mT_from_MET_{label}",
                label=r"$W^{\mathrm{PFMET}} m_T$ ",
            ).Weight(),
            f"W_pt_from_CaloMET_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"W_pt_from_CaloMET_{label}",
                label=r"$W^{\mathrm{CaloMET}}$ $p_T$",
            ).Weight(),
            f"W_pt_from_PuppiMET_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"W_pt_from_PuppiMET_{label}",
                label=r"$W^{\mathrm{PuppiMET}}$ $p_T$",
            ).Weight(),
            f"W_pt_from_MET_{label}": Hist.new.Reg(
                2000,
                0,
                2000,
                name=f"W_pt_from_MET_{label}",
                label=r"$W^{\mathrm{PFMET}}$ $p_T$",
            ).Weight(),
            f"W_phi_from_CaloMET_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"W_phi_from_CaloMET_{label}",
                label=r"$W^{\mathrm{CaloMET}}$ $\phi$",
            ).Weight(),
            f"W_phi_from_PuppiMET_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"W_phi_from_PuppiMET_{label}",
                label=r"$W^{\mathrm{PuppiMET}}$ $\phi$",
            ).Weight(),
            f"W_phi_from_MET_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"W_phi_from_MET_{label}",
                label=r"$W^{\mathrm{PFMET}}$ $\phi$",
            ).Weight(),
            f"topMass_{label}": Hist.new.Reg(
                100,
                0,
                500,
                name=f"topMass_{label}",
                label="$m_{T}$",
            ).Weight(),
            f"topMassJetClosestToMET_{label}": Hist.new.Reg(
                100,
                0,
                500,
                name=f"topMassJetClosestToMET_{label}",
                label="$m_{T}$ from jet closest to MET",
            ).Weight(),
            f"topMassBJet_{label}": Hist.new.Reg(
                100,
                0,
                500,
                name=f"topMassBJet_{label}",
                label="$m_{T}$ from highest btag jet",
            ).Weight(),
            f"deltaPhi_lepton_bjet_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_lepton_bjet_{label}",
                label=r"$\Delta\phi$($\ell$, bjet)",
            ).Weight(),
            f"deltaPhi_jet1_bjet_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_jet1_bjet_{label}",
                label=r"$\Delta\phi$(jet1, bjet)",
            ).Weight(),
            f"deltaPhi_minDeltaPhiMETJet_SUEP_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_minDeltaPhiMETJet_SUEP_{label}",
                label=r"$\Delta\phi$($\mathrm{jet}^{\mathrm{closest~to~MET}}$, SUEP)",
            ).Weight(),
            f"deltaPhi_minDeltaPhiMETJet_lepton_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_minDeltaPhiMETJet_lepton_{label}",
                label=r"$\Delta\phi$($\mathrm{jet}^{\mathrm{closest~to~MET}}$, $\ell$)",
            ).Weight(),
            f"deltaPhi_minDeltaPhiMETJet_MET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_minDeltaPhiMETJet_MET_{label}",
                label=r"$\Delta\phi$($\mathrm{jet}^{\mathrm{closest~to~MET}}$, MET)",
            ).Weight(),
            f"deltaPhi_lepton_CaloMET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_lepton_CaloMET_{label}",
                label=r"$\Delta\phi$($\ell$, CaloMET)",
            ).Weight(),
            f"deltaPhi_lepton_PuppiMET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_lepton_PuppiMET_{label}",
                label=r"$\Delta\phi$($\ell$, PuppiMET)",
            ).Weight(),
            f"deltaPhi_lepton_MET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_lepton_MET_{label}",
                label=r"$\Delta\phi$($\ell$, MET)",
            ).Weight(),
            f"deltaPhi_SUEP_CaloMET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_SUEP_CaloMET_{label}",
                label=r"$\Delta\phi$(MET, CaloMET)",
            ).Weight(),
            f"deltaPhi_SUEP_PuppiMET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_SUEP_PuppiMET_{label}",
                label=r"$\Delta\phi$(SUEP, PuppiMET)",
            ).Weight(),
            f"deltaPhi_SUEP_MET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_SUEP_MET_{label}",
                label=r"$\Delta\phi$(SUEP, PFMET)",
            ).Weight(),
            f"deltaPhi_SUEP_W_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_SUEP_W_{label}",
                label=r"$\Delta\phi$(SUEP, W)",
            ).Weight(),
            f"deltaPhi_lepton_SUEP_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_lepton_SUEP_{label}",
                label=r"$\Delta\phi$($\ell$, SUEP)",
            ).Weight(),
            f"deltaPhi_bjet_SUEP_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_bjet_SUEP_{label}",
                label=r"$\Delta\phi$(bjet, SUEP)",
            ).Weight(),
            f"deltaPhi_bjet_MET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_bjet_MET_{label}",
                label=r"$\Delta\phi$(bjet, MET)",
            ).Weight(),
            f"deltaPhi_bjet_lepton_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_bjet_lepton_{label}",
                label=r"$\Delta\phi$(bjet, lepton)",
            ).Weight(),
            f"deltaPhi_SUEP_jet1_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_SUEP_jet1_{label}",
                label=r"$\Delta\phi$(SUEP, jet1)",
            ).Weight(),
            f"deltaPhi_SUEP_bjet_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_SUEP_bjet_{label}",
                label=r"$\Delta\phi$(SUEP, bjet)",
            ).Weight(),
            f"2D_jet1_pt_vs_SUEP_nconst_{label}": Hist.new.Reg(
                1000,
                0,
                2000,
                name=f"jet1_pt_{label}",
                label="Leading AK4Jet $p_T$",
            )
            .Reg(
                501,
                0,
                500,
                name=f"nconst_{label}",
                label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
            )
            .Weight(),
            f"2D_ak4jet1_inSUEPcluster_pt_vs_SUEP_nconst_{label}": Hist.new.Reg(
                1000,
                0,
                2000,
                name=f"ak4jet1_inSUEPcluster_pt_{label}",
                label="Leading AK4Jet in SUEP AK15 cluster $p_T$",
            )
            .Reg(
                501,
                0,
                500,
                name=f"nconst_{label}",
                label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
            )
            .Weight(),
            #            f"MET_sumEt_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                5000,
            #                name=f"MET_sumEt_{label}",
            #                label="MET sumEt",
            #            ).Weight(),
            #            f"MET_JEC_pt_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"MET_JEC_pt_{label}",
            #                label="MET JEC $p_T$",
            #            ).Weight(),
            #            f"MET_JEC_pt_JER_up_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"MET_JEC_pt_JER_up_{label}",
            #                label="MET JEC $p_T$",
            #            ).Weight(),
            #            f"MET_JEC_pt_JER_down_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"MET_JEC_pt_JER_down_{label}",
            #                label="MET JEC $p_T$",
            #            ).Weight(),
            #            f"MET_JEC_pt_JES_up_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"MET_JEC_pt_JES_up_{label}",
            #                label="MET JEC $p_T$",
            #            ).Weight(),
            #            f"MET_JEC_pt_JES_down_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"MET_JEC_pt_JES_down_{label}",
            #                label="MET JEC $p_T$",
            #            ).Weight(),
            #            f"MET_JEC_pt_UnclusteredEnergy_up_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"MET_JEC_pt_UnclusteredEnergy_up_{label}",
            #                label="MET JEC $p_T$",
            #            ).Weight(),
            #            f"MET_JEC_pt_UnclusteredEnergy_down_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"MET_JEC_pt_UnclusteredEnergy_down_{label}",
            #                label="MET JCE $p_T$",
            #            ).Weight(),
            #            f"MET_JEC_phi_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_{label}",
            #                label="MET JEC $\phi$",
            #            ).Weight(),
            #            f"MET_JEC_phi_JER_up_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_JER_up_{label}",
            #                label="MET JEC $\phi$",
            #            ).Weight(),
            #            f"MET_JEC_phi_JER_down_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_JER_down_{label}",
            #                label="MET JEC $\phi$",
            #            ).Weight(),
            #            f"MET_JEC_phi_JES_up_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_JES_up_{label}",
            #                label="MET JEC $\phi$",
            #            ).Weight(),
            #            f"MET_JEC_phi_JES_down_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_JES_down_{label}",
            #                label="MET JEC $\phi$",
            #            ).Weight(),
            #            f"MET_JEC_phi_UnclusteredEnergy_up_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_UnclusteredEnergy_up_{label}",
            #                label="MET JEC $\phi$",
            #            ).Weight(),
            #            f"MET_JEC_phi_UnclusteredEnergy_down_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_UnclusteredEnergy_down_{label}",
            #                label="MET JEC $\phi$",
            #            ).Weight(),
            #            f"MET_JEC_sumEt_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                5000,
            #                name=f"MET_JEC_sumEt_{label}",
            #                label="MET JEC sumEt",
            #            ).Weight(),
        }
    )
