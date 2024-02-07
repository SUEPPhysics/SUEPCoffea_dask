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
    if "HighestPT" in label:
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
                    label="# Constituents",
                ).Weight(),
                f"{r}SUEP_S1_{label}": Hist.new.Reg(
                    100, 0, 1, name=f"{r}SUEP_S1_{label}", label="$Sph_1$"
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
                        label="# FastJets in Event",
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
                    f"2D_SUEP_S1_vs_SUEP_nconst_{label}": Hist.new.Reg(
                        100, 0, 1.0, name=f"SUEP_S1_{label}", label="$Sph_1$"
                    )
                    .Reg(501, 0, 500, name=f"nconst_{label}", label="# Constituents")
                    .Weight(),
                    f"2D_SUEP_S1_vs_ntracks_{label}": Hist.new.Reg(
                        100, 0, 1.0, name=f"SUEP_S1_{label}", label="$Sph_1$"
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
                        100,
                        0,
                        2000,
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
                        200,
                        0,
                        500,
                        name=f"{r}SUEP_pt_avg_{label}",
                        label=r"SUEP Components $p_T$ Avg.",
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
                        label="SUEP phi - ISR phi [GeV]",
                    ).Weight(),
                }
            )


def init_hists_clusterInverted(output, label, regions_list=[""]):
    output.update(
        {
            # 2D histograms
            f"2D_ISR_S1_vs_ntracks_{label}": Hist.new.Reg(
                100, 0, 1.0, name=f"ISR_S1_{label}", label="$Sph_1$"
            )
            .Reg(200, 0, 500, name=f"ntracks_{label}", label="# Tracks")
            .Weight(),
            f"2D_ISR_S1_vs_ISR_nconst_{label}": Hist.new.Reg(
                100, 0, 1.0, name=f"ISR_S1_{label}", label="$Sph_1$"
            )
            .Reg(501, 0, 500, name=f"nconst_{label}", label="# Constituents")
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
                    100, 0, 1, name=f"{r}ISR_S1_{label}", label="$Sph_1$"
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
                    100, 0, 1.0, name=f"SUEP_S1_{label}", label="$Sph_1$"
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
            .Reg(100, 0, 1, name=f"SUEP_S1_{label}", label="$Sph_1$")
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
                    label="# Constituents",
                ).Weight(),
                f"{r}SUEP_S1_{label}": Hist.new.Reg(
                    100,
                    -1,
                    2,
                    name=f"{r}SUEP_S1_{label}",
                    label="$Sph_1$",
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
                    100, 0, 1.0, name=f"ISR_S1_{label}", label="$Sph_1$"
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
            .Reg(100, 0, 1, name=f"ISR_S1_{label}", label="$Sph_1$")
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
                    100, -1, 2, name=f"{r}ISR_S1_{label}", label="$Sph_1$"
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
            f"CaloMET_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"CaloMET_pt_{label}",
                label="CaloMET pT",
            ).Weight(),
            f"CaloMET_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"CaloMET_phi_{label}",
                label="CaloMET phi",
            ).Weight(),
            f"CaloMET_sumEt_{label}": Hist.new.Reg(
                100,
                0,
                5000,
                name=f"CaloMET_sumEt_{label}",
                label="CaloMET sumEt",
            ).Weight(),
            f"PuppiMET_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"PuppiMET_pt_{label}",
                label="PuppiMET pt",
            ).Weight(),
            #            '''f"PuppiMET_pt_JER_up_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"PuppiMET_pt_JER_up_{label}",
            #                label="PuppiMET pt",
            #            ).Weight(),
            #            f"PuppiMET_pt_JER_down_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"PuppiMET_pt_JER_down_{label}",
            #                label="PuppiMET pt",
            #            ).Weight(),
            #            f"PuppiMET_pt_JES_up_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"PuppiMET_pt_JES_up{label}",
            #                label="PuppiMET pt",
            #            ).Weight(),
            #            f"PuppiMET_pt_JES_down_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"PuppiMET_pt_JES_down{label}",
            #                label="PuppiMET pt",
            #            ).Weight(),'''
            f"PuppiMET_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"PuppiMET_phi_{label}",
                label="PuppiMET phi",
            ).Weight(),
            #            f"PuppiMET_phi_JER_up_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"PuppiMET_phi_JER_up_{label}",
            #                label="PuppiMET phi",
            #            ).Weight(),
            #            f"PuppiMET_phi_JER_down_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"PuppiMET_phi_JER_down_{label}",
            #                label="PuppiMET phi",
            #            ).Weight(),
            #            f"PuppiMET_phi_JES_up_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"PuppiMET_phi_JES_up_{label}",
            #                label="PuppiMET phi",
            #            ).Weight(),
            #            f"PuppiMET_phi_JES_down_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"PuppiMET_phi_JES_down_{label}",
            #                label="PuppiMET phi",
            #            ).Weight(),'''
            f"PuppiMET_sumEt_{label}": Hist.new.Reg(
                100,
                0,
                5000,
                name=f"PuppiMET_sumEt_{label}",
                label="PuppiMET sumEt",
            ).Weight(),
            f"MET_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"MET_pt_{label}",
                label="MET pt",
            ).Weight(),
            f"MET_phi_{label}": Hist.new.Reg(
                100,
                -4,
                4,
                name=f"MET_phi_{label}",
                label="MET phi",
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
            f"jet1_pT_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"jet1_pT_{label}",
                label="highest pT jet pT",
            ).Weight(),
            f"jet1_phi_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"jet1_phi_{label}",
                label="highest pT jet phi",
            ).Weight(),
            f"jet1_eta_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"jet1_eta_{label}",
                label="highest pT jet eta",
            ).Weight(),
            f"jet1_qgl_{label}": Hist.new.Reg(
                20,
                0,
                1,
                name=f"jet1_qgl_{label}",
                label="highest pT jet qgl",
            ).Weight(),
            f"jet2_pT_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"jet2_pT_{label}",
                label="2nd highest pT jet pT",
            ).Weight(),
            f"jet2_phi_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"jet2_phi_{label}",
                label="2nd highest pT jet phi",
            ).Weight(),
            f"jet2_eta_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"jet2_eta_{label}",
                label="2nd highest pT jet eta",
            ).Weight(),
            f"jet2_qgl_{label}": Hist.new.Reg(
                20,
                0,
                1,
                name=f"jet2_qgl_{label}",
                label="2nd highest pT jet qgl",
            ).Weight(),
            f"jet3_pT_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"jet3_pT_{label}",
                label="3rd highest pT jet pT",
            ).Weight(),
            f"jet3_phi_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"jet3_phi_{label}",
                label="3rd highest pT jet phi",
            ).Weight(),
            f"jet3_eta_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"jet3_eta_{label}",
                label="3rd highest pT jet eta",
            ).Weight(),
            f"jet3_qgl_{label}": Hist.new.Reg(
                20,
                0,
                1,
                name=f"jet3_qgl_{label}",
                label="3rd highest pT jet qgl",
            ).Weight(),
            f"highest_btag_jet_pt_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"highest_btag_jet_pt_{label}",
                label="highest btag jet pT",
            ).Weight(),
            f"highest_btag_jet_phi_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"highest_btag_jet_phi_{label}",
                label="highest btag jet phi",
            ).Weight(),
            f"highest_btag_jet_eta_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"highest_btag_jet_eta_{label}",
                label="highest btag jet eta",
            ).Weight(),
            f"highest_btag_jet_qgl_{label}": Hist.new.Reg(
                20,
                0,
                1,
                name=f"highest_btag_jet_qgl_{label}",
                label="highest btag jet qgl",
            ).Weight(),
            f"lepton_pt_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"lepton_pt_{label}",
                label="lepton pT",
            ).Weight(),
            f"lepton_phi_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"lepton_phi_{label}",
                label="lepton phi",
            ).Weight(),
            f"lepton_eta_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"lepton_eta_{label}",
                label="lepton eta",
            ).Weight(),
            f"lepton_mass_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"lepton_mass_{label}",
                label="lepton mass",
            ).Weight(),
            f"lepton_flavor_{label}": Hist.new.Reg(
                40,
                -20,
                20,
                name=f"lepton_flavor_{label}",
                label="lepton pdgID",
            ).Weight(),
            f"lepton_ID_{label}": Hist.new.Reg(
                10,
                0,
                10,
                name=f"lepton_ID_{label}",
                label="lepton ID",
            ).Weight(),
            f"lepton_IDMVA_{label}": Hist.new.Reg(
                10,
                0,
                10,
                name=f"lepton_IDMVA_{label}",
                label="lepton IDMVA",
            ).Weight(),
            f"lepton_iso_{label}": Hist.new.Reg(
                50,
                0,
                1,
                name=f"lepton_iso_{label}",
                label="lepton iso",
            ).Weight(),
            f"lepton_isoMVA_{label}": Hist.new.Reg(
                50,
                0,
                1,
                name=f"lepton_isoMVA_{label}",
                label="lepton isoMVA",
            ).Weight(),
            f"lepton_dxy_{label}": Hist.new.Reg(
                100,
                -0.2,
                0.2,
                name=f"lepton_dxy_{label}",
                label="lepton dxy",
            ).Weight(),
            f"lepton_dz_{label}": Hist.new.Reg(
                100,
                -0.2,
                0.2,
                name=f"lepton_dz_{label}",
                label="lepton dz",
            ).Weight(),
            f"W_mT_from_CaloMET_{label}": Hist.new.Reg(
                201,
                0,
                200,
                name=f"W_mT_from_CaloMET_{label}",
                label=" (from CaloMET) W mT ",
            ).Weight(),
            f"W_mT_from_PuppiMET_{label}": Hist.new.Reg(
                201,
                0,
                200,
                name=f"W_mT_from_PuppiMET_{label}",
                label=" (from PuppiMET) W mT ",
            ).Weight(),
            f"W_mT_from_MET_{label}": Hist.new.Reg(
                201,
                0,
                200,
                name=f"W_mT_from_MET_{label}",
                label=" (from PFMET) W mT ",
            ).Weight(),
            f"W_pT_from_CaloMET_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"W_pT_from_CaloMET_{label}",
                label="(from CaloMET) W pT",
            ).Weight(),
            f"W_pT_from_PuppiMET_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"W_pT_from_PuppiMET_{label}",
                label="(from PuppiMET) W pT",
            ).Weight(),
            f"W_pT_from_MET_{label}": Hist.new.Reg(
                100,
                0,
                2000,
                name=f"W_pT_from_MET_{label}",
                label="(from PF MET) W pT",
            ).Weight(),
            f"W_phi_from_CaloMET_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"W_phi_from_CaloMET_{label}",
                label="(from CaloMET) W phi",
            ).Weight(),
            f"W_phi_from_PuppiMET_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"W_phi_from_PuppiMET_{label}",
                label="(from PuppiMET) W phi",
            ).Weight(),
            f"W_phi_from_MET_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"W_phi_from_MET_{label}",
                label="(from PF MET) W phi",
            ).Weight(),
            f"deltaPhi_lepton_CaloMET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_lepton_CaloMET_{label}",
                label="deltaPhi lepton CaloMET",
            ).Weight(),
            f"deltaPhi_lepton_PuppiMET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_lepton_PuppiMET_{label}",
                label="deltaPhi lepton PuppiMET",
            ).Weight(),
            f"deltaPhi_lepton_MET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_lepton_MET_{label}",
                label="deltaPhi lepton PFMET",
            ).Weight(),
            f"deltaPhi_lepton_CaloMET_func_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_lepton_CaloMET_func_{label}",
                label="deltaPhi lepton CaloMET (func)",
            ).Weight(),
            f"deltaPhi_lepton_MET_func_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_lepton_MET_func_{label}",
                label="deltaPhi lepton PFMET (func)",
            ).Weight(),
            f"deltaPhi_lepton_MET_func_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_lepton_MET_func_{label}",
                label="deltaPhi lepton PFMET (func)",
            ).Weight(),
            f"deltaPhi_SUEP_CaloMET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_SUEP_CaloMET_{label}",
                label="deltaPhi SUEP CaloMET",
            ).Weight(),
            f"deltaPhi_SUEP_PuppiMET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_SUEP_PuppiMET_{label}",
                label="deltaPhi SUEP PuppiMET",
            ).Weight(),
            f"deltaPhi_SUEP_MET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_SUEP_MET_{label}",
                label="deltaPhi SUEP PFMET",
            ).Weight(),
            f"2D_MET_pt_vs_W_mT_from_MET_{label}": Hist.new.Reg(
                100, 0, 1000, name=f"MET_pt_{label}", label="MET pt"
            )
            .Reg(200, 0, 200, name=f"W_mT_from_MET_{label}", label="(from PFMET) W mT ")
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
            #                label="MET JEC pt",
            #            ).Weight(),
            #            f"MET_JEC_pt_JER_up_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"MET_JEC_pt_JER_up_{label}",
            #                label="MET JEC pt",
            #            ).Weight(),
            #            f"MET_JEC_pt_JER_down_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"MET_JEC_pt_JER_down_{label}",
            #                label="MET JEC pt",
            #            ).Weight(),
            #            f"MET_JEC_pt_JES_up_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"MET_JEC_pt_JES_up_{label}",
            #                label="MET JEC pt",
            #            ).Weight(),
            #            f"MET_JEC_pt_JES_down_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"MET_JEC_pt_JES_down_{label}",
            #                label="MET JEC pt",
            #            ).Weight(),
            #            f"MET_JEC_pt_UnclusteredEnergy_up_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"MET_JEC_pt_UnclusteredEnergy_up_{label}",
            #                label="MET JEC pt",
            #            ).Weight(),
            #            f"MET_JEC_pt_UnclusteredEnergy_down_{label}": Hist.new.Reg(
            #                100,
            #                0,
            #                1000,
            #                name=f"MET_JEC_pt_UnclusteredEnergy_down_{label}",
            #                label="MET JCE pt",
            #            ).Weight(),
            #            f"MET_JEC_phi_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_{label}",
            #                label="MET JEC phi",
            #            ).Weight(),
            #            f"MET_JEC_phi_JER_up_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_JER_up_{label}",
            #                label="MET JEC phi",
            #            ).Weight(),
            #            f"MET_JEC_phi_JER_down_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_JER_down_{label}",
            #                label="MET JEC phi",
            #            ).Weight(),
            #            f"MET_JEC_phi_JES_up_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_JES_up_{label}",
            #                label="MET JEC phi",
            #            ).Weight(),
            #            f"MET_JEC_phi_JES_down_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_JES_down_{label}",
            #                label="MET JEC phi",
            #            ).Weight(),
            #            f"MET_JEC_phi_UnclusteredEnergy_up_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_UnclusteredEnergy_up_{label}",
            #                label="MET JEC phi",
            #            ).Weight(),
            #            f"MET_JEC_phi_UnclusteredEnergy_down_{label}": Hist.new.Reg(
            #                100,
            #                -4,
            #                4,
            #                name=f"MET_JEC_phi_UnclusteredEnergy_down_{label}",
            #                label="MET JEC phi",
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
