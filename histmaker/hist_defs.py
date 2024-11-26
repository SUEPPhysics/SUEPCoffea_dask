import numpy as np
from hist import Hist


def initialize_histograms(output: dict, label: str, options, config: dict) -> dict:
    # don't recreate histograms if called multiple times with the same output label
    if label in output.get("labels", []):
        return output
    elif "labels" in output.keys():
        output["labels"].append(label)

    regions_list = [""]
    if options.doABCD:
        init_hists_ABCD(output, label, config)
        regions_list += get_ABCD_regions(config)

    ###########################################################################################################################
    if "Cluster" in label:
        init_hists_cluster(output, label, regions_list)

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
    # WH analysis
    if 'WH' in options.channel:
        if options.limits:
            init_hists_WHlimits(output, label, regions_list)
        else:
            # common to WH and WH-VRGJ
            init_hists_highestPT(output, label, options)

            # specific stuff for each region
            if 'WH' == options.channel:
                init_hists_WH(output, label)
            elif 'VRGJ' == options.channel:
                init_hists_VRGJ(output, label)

    return output


def get_ABCD_regions(config: dict) -> list:
    xvar_regions = config["xvar_regions"]
    yvar_regions = config["yvar_regions"]
    regions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n_regions = (len(xvar_regions) - 1) * (len(yvar_regions) - 1)
    return [regions[i] + "_" for i in range(n_regions)]


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


def init_hists_cluster(output, label, regions_list=[""]):

    for r in regions_list:

        # these histograms will be made for each systematic, and each ABCD region
        output.update(
            {
                f"{r}SUEP_nconst_{label}": Hist.new.Variable(
                    np.linspace(-0.5, 499.5, 501),
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
                f"{r}SUEP_pt_{label}": Hist.new.Reg(
                    1000,
                    0,
                    1000,
                    name=f"{r}SUEP_pt_{label}",
                    label=r"$p^{\mathrm{SUEP}}_T$ [GeV]",
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
                    f"ntracks_{label}": Hist.new.Variable(
                        np.linspace(-0.5, 499.5, 501),
                        name=f"ntracks_{label}",
                        label=r"$n^{\mathrm{event}}_{\mathrm{tracks}}$",
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
                    f"2D_SUEP_S1_vs_SUEP_nconst_{label}": Hist.new.Reg(
                        100,
                        0,
                        1.0,
                        name=f"SUEP_S1_{label}",
                        label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
                    )
                    .Variable(
                        np.linspace(-0.5, 499.5, 501),
                        name=f"nconst_{label}",
                        label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
                    )
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
                    f"{r}SUEP_delta_pt_genPt_{label}": Hist.new.Reg(
                        100,
                        -2000,
                        2000,
                        name=f"{r}SUEP_delta_pt_genPt_{label}",
                        label="SUEP $p_T$ - genSUEP $p_T$ [GeV]",
                    ).Weight(),
                    f"{r}SUEP_ISR_deltaPhi_{label}": Hist.new.Reg(
                        100,
                        -6,
                        6,
                        name=f"{r}SUEP_ISR_deltaPhi_{label}",
                        label=r"SUEP $\phi$ - ISR $\phi$ [GeV]",
                    ).Weight(),
                    f"{r}SUEP_pt_avg_{label}": Hist.new.Reg(
                        100,
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
                        100,
                        0,
                        2000,
                        name=f"{r}SUEP_mass_{label}",
                        label="SUEP Mass [GeV]",
                    ).Weight(),
                    f"{r}SUEP_delta_mass_genMass_{label}": Hist.new.Reg(
                        100,
                        -2000,
                        2000,
                        name=f"{r}SUEP_delta_mass_genMass_{label}",
                        label="SUEP Mass - genSUEP Mass [GeV]",
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
                500,
                0,
                500,
                name=f"nconst_{label}",
                label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
            )
            .Weight(),
            f"2D_ISR_nconst_vs_ISR_pt_avg_{label}": Hist.new.Reg(
                500, 0, 500, name=f"ISR_nconst_{label}"
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
                    500,
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
                f"2D_SUEP_nconst_vs_{model}_{label}": Hist.new.Variable(
                    np.linspace(-0.5, 499.5, 501),
                    name=f"SUEP_nconst_{label}",
                    label="# Const",
                )
                .Reg(100, 0, 1, name=f"{model}_{label}", label="GNN Output")
                .Weight(),
            }
        )

    for r in regions_list:
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
                    500, 0, 500, name=f"ISR_nconst_{label}", label="# Const"
                )
                .Reg(100, 0, 1, name=f"{model}_{label}", label="GNN Output")
                .Weight(),
            }
        )
    output.update(
        {
            f"2D_ISR_nconst_vs_ISR_S1_{label}": Hist.new.Reg(
                500, 0, 500, name=f"ISR_nconst_{label}", label="# Const"
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
                    500,
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


def init_hists_highestPT(output, label, options, regions_list=[""]):
    output.update(
        {
            f"event_weight_{label}": Hist.new.Reg(
                1000,
                -500,
                500,
                name=f"event_weight_{label}",
                label="Event Weight",
            ).Weight(),
            f"SUEP_nconst_{label}": Hist.new.Variable(
                np.linspace(-0.5, 199.5, 201),
                name=f"SUEP_nconst_{label}",
                label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
            ).Weight(),
            f"SUEP_S1_{label}": Hist.new.Reg(
                1000,
                0,
                1,
                name=f"SUEP_S1_{label}",
                label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
            ).Weight(),
            f"SUEP_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"SUEP_pt_{label}",
                label=r"$p^{\mathrm{SUEP}}_T$ [GeV]",
            ).Weight(),
            f"otherAK15_maxPT_pt_{label}": Hist.new.Reg(
                1000,
                0,
                200,
                name=f"otherAK15_maxPT_pt_{label}",
                label=r"$p^{\mathrm{sub-leading~AK15}}_T$ [GeV]",
            ).Weight(),
            f"ntracks_{label}": Hist.new.Variable(
                np.linspace(-0.5, 99.5, 101),
                name=f"ntracks_{label}",
                label=r"$n^{\mathrm{event}}_{\mathrm{tracks}}$",
            ).Weight(),
            f"npfcands_{label}": Hist.new.Variable(
                np.linspace(-0.5, 99.5, 101),
                name=f"npfcands_{label}",
                label=r"$n^{\mathrm{event}}_{\mathrm{pfcands}}$",
            ).Weight(),
            f"nlosttracks_{label}": Hist.new.Variable(
                np.linspace(-0.5, 99.5, 101),
                name=f"nlosttracks_{label}",
                label=r"$n^{\mathrm{event}}_{\mathrm{lost tracks}}$",
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
            f"2D_SUEP_S1_vs_SUEP_nconst_{label}": Hist.new.Reg(
                100,
                0,
                1.0,
                name=f"SUEP_S1_{label}",
                label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
            )
            .Variable(
                np.linspace(-0.5, 199.5, 201),
                name=f"nconst_{label}",
                label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
            )
            .Weight(),
            f"3D_SUEP_S1_vs_SUEP_nconst_vs_SUEP_pt_{label}": Hist.new.Reg(
                100,
                0,
                1.0,
                name=f"SUEP_S1_{label}",
                label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
            )
            .Variable(
                np.linspace(-0.5, 99.5, 100),
                name=f"SUEP_nconst_{label}",
                label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
            )
            .Reg(
                100,
                0,
                500,
                name=f"SUEP_pt_{label}",
                label=r"$p^{\mathrm{SUEP}}_T$ [GeV]",
            )
            .Weight(),
            f"SUEP_pt_avg_{label}": Hist.new.Reg(
                500,
                0,
                100,
                name=f"SUEP_pt_avg_{label}",
                label="SUEP Components $p_T$ Avg.",
            ).Weight(),
            f"SUEP_pt_avg_b_{label}": Hist.new.Reg(
                100,
                0,
                20,
                name=f"SUEP_pt_avg_b_{label}",
                label=r"SUEP Components $p_T^{\mathrm{boosted}}$ Avg.",
            ).Weight(),
            f"SUEP_eta_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"SUEP_eta_{label}",
                label=r"SUEP $\eta$",
            ).Weight(),
            f"SUEP_phi_{label}": Hist.new.Reg(
                100,
                -6.5,
                6.5,
                name=f"SUEP_phi_{label}",
                label=r"SUEP $\phi$",
            ).Weight(),
            f"SUEP_mass_{label}": Hist.new.Reg(
                150,
                0,
                2000,
                name=f"SUEP_mass_{label}",
                label="SUEP Mass [GeV]",
            ).Weight(),
            f"ht_JEC_{label}": Hist.new.Reg(
                100, 0, 10000, name=f"ht_JEC_{label}", label="HT"
            ).Weight(),  
            f"WH_MET_pt_{label}": Hist.new.Reg(
                200,
                0,
                1000,
                name=f"WH_MET_pt_{label}",
                label=r"$p^{\mathrm{WH MET}}_T$ [GeV]",
            ).Weight(),
            f"WH_MET_phi_{label}": Hist.new.Reg(
                100,
                -3,
                3,
                name=f"WH_MET_phi_{label}",
                label=r"$\phi^{\mathrm{WH MET}}$ [GeV]",
            ).Weight(),
            f"nBLoose_{label}": Hist.new.Reg(
                20,
                0,
                20,
                name=f"nBLoose_{label}",
                label=r"$n_{\mathrm{Loose~B-jets}}$",
            ).Weight(),
            # f"nBMedium_{label}": Hist.new.Reg(
            #     20,
            #     0,
            #     20,
            #     name=f"nBMedium_{label}",
            #     label="$n_{\mathrm{Medium~B-jets}$",
            # ).Weight(),
            f"nBTight_{label}": Hist.new.Reg(
                20,
                0,
                20,
                name=f"nBTight_{label}",
                label=r"$n_{\mathrm{Tight~B-jets}}$",
            ).Weight(),
            f"ak4jets_outsideSUEPcluster_pt_{label}": Hist.new.Reg(
                100,
                0,
                500,
                name=f"ak4jets_outsideSUEPcluster_pt_{label}",
                label="ak4jets outside SUEP cluster $p_T$",
            ).Weight(),
            f"jet1_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"jet1_pt_{label}",
                label=r"$p^{\mathrm{jet}}_T$",
            ).Weight(),
            f"jet2_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"jet2_pt_{label}",
                label=r"$p^{\mathrm{jet}}_T$",
            ).Weight(),
            f"jet1_qgl_{label}": Hist.new.Reg(
                100,
                0,
                1,
                name=f"jet1_qgl_{label}",
                label=r"$qgl^{\mathrm{jet}}$",
            ).Weight(),
            f"jet2_qgl_{label}": Hist.new.Reg(
                100,
                0,
                1,
                name=f"jet2_qgl_{label}",
                label=r"$qgl^{\mathrm{jet}}$",
            ).Weight(),
            f"bjetSel_{label}": Hist.new.Int(
                0,
                2,
                name=f"bjetSel_{label}",
                label="b-jet selection",
            ).Weight(),
            f"ak4jets_inSUEPcluster_n_{label}": Hist.new.Reg(
                20,
                0,
                20,
                name=f"ak4jets_inSUEPcluster_n_{label}",
                label="# ak4jets in SUEP cluster",
            ).Weight(),
            f"ak4jets_outsideSUEPcluster_n_{label}": Hist.new.Reg(
                20,
                0,
                20,
                name=f"ak4jets_oustideSUEPcluster_n_{label}",
                label="# ak4jets oustide SUEP cluster",
            ).Weight(),
            f"deltaPhi_ak4jet1_inSUEPcluster_SUEP_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_ak4jet1_inSUEPcluster_SUEP_{label}",
                label=r"$\Delta\phi(\mathrm{ak4jet1, SUEP})$",
            ).Weight(),
            f"ak4jet1_inSUEPcluster_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"ak4jet1_inSUEPcluster_pt_{label}",
                label="ak4jet1 in SUEP cluster $p_T$",
            ).Weight(),
            f"ak4jet2_inSUEPcluster_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"ak4jet2_inSUEPcluster_pt_{label}",
                label="ak4jet2 in SUEP cluster $p_T$",
            ).Weight(),
            f"ak4jet1_inSUEPcluster_eta_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"ak4jet1_inSUEPcluster_eta_{label}",
                label=r"ak4jet1 in SUEP cluster $\eta$",
            ).Weight(),
            f"deltaPhi_ak4jet1_outsideSUEPcluster_SUEP_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_ak4jet1_outsideSUEPcluster_SUEP_{label}",
                label=r"$\Delta\phi(\mathrm{ak4jet1, SUEP})$",
            ).Weight(),
            f"ak4jet1_outsideSUEPcluster_pt_{label}": Hist.new.Reg(
                100,
                0,
                500,
                name=f"ak4jet1_outsideSUEPcluster_pt_{label}",
                label="ak4jet1 outside SUEP cluster $p_T$",
            ).Weight(),
            f"ak4jet1_outsideSUEPcluster_eta_{label}": Hist.new.Reg(
                100,
                -5,
                5,
                name=f"ak4jet1_outsideSUEPcluster_eta_{label}",
                label=r"ak4jet1 outside SUEP cluster $\eta$",
            ).Weight(),         
            # f"minDeltaPhiMETJet_pt_{label}": Hist.new.Reg(
            #     100,
            #     0,
            #     2000,
            #     name=f"minDeltaPhiMETJet_pt_{label}",
            #     label=r"$\mathrm{jet}^{\mathrm{closest~to~MET}}$ $p_T$",
            # ).Weight(),
            # f"minDeltaPhiMETJet_phi_{label}": Hist.new.Reg(
            #     60,
            #     -3.2,
            #     3.2,
            #     name=f"minDeltaPhiMETJet_phi_{label}",
            #     label=r"$\mathrm{jet}^{\mathrm{closest~to~MET}}$ $\phi$",
            # ).Weight(),
            f"nLooseLeptons_{label}": Hist.new.Reg(
                10,
                0,
                10,
                name=f"nLooseLeptons_{label}",
                label="# loose leptons",
            ).Weight(),
            f"nphotons_{label}": Hist.new.Int(
                0,
                10,
                name=f"nphotons_{label}",
                label=r"$n_\gamma$",
            ).Weight(),
            f"deltaPhi_minDeltaPhiMETJet_MET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_minDeltaPhiMETJet_MET_{label}",
                label=r"$\Delta\phi$($\mathrm{jet}^{\mathrm{closest~to~MET}}$, MET)",
            ).Weight(),
            f"deltaPhi_SUEP_MET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_SUEP_MET_{label}",
                label=r"$\Delta\phi$(SUEP, MET)",
            ).Weight(),
            
            f"deltaPhi_lepton_SUEP_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_lepton_SUEP_{label}",
                label=r"$\Delta\phi$($\ell$, SUEP)",
            ).Weight(),      
            f"2D_SUEP_pt_vs_SUEP_nconst_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"SUEP_pt_{label}",
                label="SUEP $p_T$ [GeV]",
            )
            .Variable(
                np.linspace(-0.5, 199.5, 201),
                name=f"SUEP_nconst_{label}",
                label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
            ).Weight(),
        }
    )

    # if options.isMC:

    #     output.update({
    #         # f"n_darkphis_{label}": Hist.new.Reg(
    #         #     200,
    #         #     0,
    #         #     200,
    #         #     name=f"n_darkphis_{label}",
    #         #     label="$n_{\phi_D}"
    #         # ).Weight(),
    #         # f"n_darkphis_inTracker_{label}": Hist.new.Reg(
    #         #     200,
    #         #     0,
    #         #     200,
    #         #     name=f"n_darkphis_inTracker_{label}",
    #         #     label="$n_{\phi_D}$ in Tracker ($|\eta| < 2.5$)"
    #         # ).Weight(),
    #         # f"percent_darkphis_inTracker_{label}": Hist.new.Reg(
    #         #     100,
    #         #     0,
    #         #     1.1,
    #         #     name=f"percent_darkphis_inTracker_{label}",
    #         #     label="% $\phi_D$ in Tracker ($|\eta| < 2.5$)"
    #         # ).Weight(),
    #         f"SUEP_genMass_{label}": Hist.new.Reg(
    #             100,
    #             0,
    #             1200,
    #             name=f"SUEP_genMass_{label}",
    #             label="Gen Mass of SUEP ($m_S$) [GeV]",
    #         ).Weight(),
    #         f"SUEP_delta_mass_genMass_{label}": Hist.new.Reg(
    #             400,
    #             -2000,
    #             2000,
    #             name=f"SUEP_delta_mass_genMass_{label}",
    #             label="SUEP Mass - genSUEP Mass [GeV]",
    #         ).Weight(),
    #         f"bTagWeight_central_wh_{label}": Hist.new.Reg(
    #             100,
    #             0,
    #             2,
    #             name=f"bTagWeight_central_wh_{label}",
    #             label="b-tag weight",
    #         ).Weight(),
    #         f"2D_bTagWeight_central_wh_vs_SUEP_nconst_{label}": Hist.new.Reg(
    #             100,
    #             0,
    #             2,
    #             name=f"bTagWeight_central_wh_{label}",
    #             label="b-tag weight",
    #         )
    #         .Var(
    #             np.linspace(-0.5, 199.5, 201),
    #             name=f"SUEP_nconst_{label}",
    #             label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
    #         )
    #         .Weight(),
    #         f"W_genW_BV_{label}": Hist.new.Reg(
    #             100,
    #             -1,
    #             2,
    #             name=f"W_genW_BV_{label}",
    #             label="($p_T^W - p_T^{genW}$)/$p_T^{genW}$",
    #         ).Weight(),
    #         f"sumAK4W_pt_{label}": Hist.new.Reg(
    #             200,
    #             0,
    #             1000,
    #             name=f"sumAK4W_pt_{label}",
    #             label=r'|$\Sigma_{i=0}\vec{p}^{\mathrm{ak4jet,~i}}_T + \vec{p}^{W}_T$|',
    #         ).Weight(),
    #         f"sumAK4W_W_BV_{label}": Hist.new.Reg(
    #             100,
    #             -1,
    #             2,
    #             name=f"sumAK4W_W_BV_{label}",
    #             label=r'$\Sigma_{i=0}\vec{p}^{\mathrm{ak4jet,~i}}_T + \vec{p}^{W}_T$ / |$\vec{p}^{W}_T$|',
    #         ).Weight(),
    #         f"genW_pt_{label}": Hist.new.Reg(
    #             200,
    #             0,
    #             2000,
    #             name=f"genW_pt_{label}",
    #             label=r"$p^{\mathrm{genW}}_T$",
    #         ).Weight(),
    #         f"genW_phi_{label}": Hist.new.Reg(
    #             60,
    #             -3.2,
    #             3.2,
    #             name=f"genW_phi_{label}",
    #             label=r"$\phi^{\mathrm{genW}}$",
    #         ).Weight(),
    #         f"genW_mass_{label}": Hist.new.Reg(
    #             100,
    #             30,
    #             130,
    #             name=f"genW_mass_{label}",
    #             label="genW Mass [GeV]",
    #         ).Weight(),
    #         f"4D_jets_pt_vs_jets_eta_vs_jets_hadronFlavor_vs_jets_btag_category_{label}": Hist.new.Variable(
    #             [0, 30, 50, 70, 100, 140, 200, 300, 600, 1000], name="jets_pt"
    #         )
    #         .Variable([0, 1.44, 2.5], name="jets_eta")
    #         .IntCategory([0, 4, 5], name="jets_hadronFlavor")
    #         .Variable([0, 1, 2, 3], name="jets_btag_category")
    #         .Weight(),  
    #         f"deltaR_genSUEP_SUEP_{label}": Hist.new.Reg(
    #             100,
    #             0,
    #             6,
    #             name=f"deltaR_genSUEP_SUEP_{label}",
    #             label=r"$\Delta R$(gen. SUEP, reco. SUEP)",
    #         ).Weight(),
    #         f"SUEP_genSUEP_BV_{label}": Hist.new.Reg(
    #             100,
    #             -1,
    #             2,
    #             name=f"genSUEP_SUEP_BV_{label}",
    #             label=r"($p^{\mathrm{reco. SUEP}}_T$ - $p^{\mathrm{gen. SUEP}}_T$) / $p^{\mathrm{gen. SUEP}}_T$",
    #         ).Weight(),
    #         f"LepSF_{label}": Hist.new.Reg(
    #             100,
    #             0.8,
    #             1.2,
    #             name=f"LepSF_{label}",
    #             label="LepSF",
    #         ).Weight(),
    #         f"LepSFMu_{label}": Hist.new.Reg(
    #             100,
    #             0.8,
    #             1.2,
    #             name=f"LepSFMu_{label}",
    #             label=r"LepSF ($\mu$)",
    #         ).Weight(),
    #         f"LepSFEl_{label}": Hist.new.Reg(
    #             100,
    #             0.8,
    #             1.2,
    #             name=f"LepSFEl_{label}",
    #             label=r"LepSF ($e$)",
    #         ).Weight(),
    #         f"LepSFMuUp_{label}": Hist.new.Reg(
    #             100,
    #             0.8,
    #             1.2,
    #             name=f"LepSFMuUp_{label}",
    #             label=r"LepSFMuUp",
    #         ).Weight(),
    #         f"LepSFMuDown_{label}": Hist.new.Reg(
    #             100,
    #             0.8,
    #             1.2,
    #             name=f"LepSFMuDown_{label}",
    #             label=r"LepSFMuDown",
    #         ).Weight(),
    #         f"LepSFEleUp_{label}": Hist.new.Reg(
    #             100,
    #             0.8,
    #             1.2,
    #             name=f"LepSFEleUp_{label}",
    #             label=r"LepSFEleUp",
    #         ).Weight(),
    #         f"LepSFEleDown_{label}": Hist.new.Reg(
    #             100,
    #             0.8,
    #             1.2,
    #             name=f"LepSFEleDown_{label}",
    #             label=r"LepSFEleDown",
    #         ).Weight(),
    #    })


def init_hists_WH(output, label):

    output.update({
        f"lepton_pt_{label}": Hist.new.Reg(
                500,
                0,
                500,
                name=f"lepton_pt_{label}",
                label=r"$p_T^{\ell}$ [GeV]",
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
            # f"lepton_mass_{label}": Hist.new.Reg(
            #     100,
            #     -5,
            #     5,
            #     name=f"lepton_mass_{label}",
            #     label=r"$m^{\ell}$",
            # ).Weight(),
            # f"lepton_flavor_{label}": Hist.new.Reg(
            #     40,
            #     -20,
            #     20,
            #     name=f"lepton_flavor_{label}",
            #     label=r"$\ell$ pdgID",
            # ).Weight(),
            # f"lepton_ID_{label}": Hist.new.Reg(
            #     10,
            #     0,
            #     10,
            #     name=f"lepton_ID_{label}",
            #     label=r"$\ell$ ID",
            # ).Weight(),
            # f"lepton_IDMVA_{label}": Hist.new.Reg(
            #     10,
            #     0,
            #     10,
            #     name=f"lepton_IDMVA_{label}",
            #     label=r"$\ell$ IDMVA",
            # ).Weight(),
            f"lepton_iso_{label}": Hist.new.Reg(
                50,
                0,
                1,
                name=f"lepton_iso_{label}",
                label=r"$\ell$ iso",
            ).Weight(),
            # f"lepton_miniIso_{label}": Hist.new.Reg(
            #     50,
            #     0,
            #     1,
            #     name=f"lepton_iso_{label}",
            #     label=r"$\ell$ miniIso",
            # ).Weight(),
            # f"lepton_isoMVA_{label}": Hist.new.Reg(
            #     50,
            #     0,
            #     1,
            #     name=f"lepton_isoMVA_{label}",
            #     label=r"$\ell$ isoMVA",
            # ).Weight(),
            # f"lepton_dxy_{label}": Hist.new.Reg(
            #     100,
            #     -0.2,
            #     0.2,
            #     name=f"lepton_dxy_{label}",
            #     label=r"$\ell$ dxy",
            # ).Weight(),
            # f"lepton_dz_{label}": Hist.new.Reg(
            #     100,
            #     -0.2,
            #     0.2,
            #     name=f"lepton_dz_{label}",
            #     label=r"$\ell$ dz",
            # ).Weight(),
            f"deltaPhi_lepton_MET_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_lepton_MET_{label}",
                label=r"$\Delta\phi$($\ell$, MET)",
            ).Weight(),
            f"W_SUEP_BV_{label}": Hist.new.Reg(
                100,
                -1,
                2,
                name=f"W_SUEP_BV_{label}",
                label="($p_T^W - p_T^{SUEP}$)/$p_T^{SUEP}$",
            ).Weight(),
            f"ak4jet1_inSUEPcluster_W_BV_{label}": Hist.new.Reg(
                100,
                -1,
                2,
                name=f"ak4jet1_inSUEPcluster_W_BV_{label}",
                label=r"($p_T^{ak4jet1} - p_T^{W}$)/$p_T^{W}$",
            ).Weight(),
            f"W_ak4jet1_inSUEPcluster_vBV_{label}": Hist.new.Reg(
                100,
                -1,
                2,
                name=f"W_ak4jet1_inSUEPcluster_vBV_{label}",
                label=r"|$\vec{p}_T^{W} + \vec{p}_T^{ak4jet1}$| / $p_T^{ak4jet1}$",
            ).Weight(),
            f"W_mt_{label}": Hist.new.Reg(
                200,
                0,
                200,
                name=f"W_mt_{label}",
                label=r"$m^{\mathrm{W}}_T$ [GeV]",
            ).Weight(),
            f"W_pt_{label}": Hist.new.Reg(
                200,
                0,
                2000,
                name=f"W_pt_{label}",
                label=r"$p^{\mathrm{W}}_T$ [GeV]",
            ).Weight(),
            f"W_pt_JER_down_{label}": Hist.new.Reg(
                200,
                0,
                2000,
                name=f"W_pt_JER_down_{label}",
                label=r"$p^{\mathrm{W}}_T$ [GeV]",
            ).Weight(),
            f"W_mt_JER_down_{label}": Hist.new.Reg(
                200,
                0,
                200,
                name=f"W_mt_JER_down_{label}",
                label=r"$m^{\mathrm{W}}_T$ [GeV]",
            ).Weight(),
            f"PuppiMET_pt_{label}": Hist.new.Reg(
                200,
                0,
                1000,
                name=f"PuppiMET_pt_{label}",
                label=r"$p^{\mathrm{PuppiMET}}_T$ [GeV]",
            ).Weight(),
            f"PuppiMET_pt_JER_down_{label}": Hist.new.Reg(
                200,
                0,
                1000,
                name=f"PuppiMET_pt_JER_down_{label}",
                label=r"$p^{\mathrm{PuppiMET}}_T$ [GeV]",
            ).Weight(),
            f"PuppiMET_phi_{label}": Hist.new.Reg(
                100,
                -3,
                3,
                name=f"PuppiMET_phi_{label}",
                label=r"$\phi^{\mathrm{PuppiMET}}$ [GeV]",
            ).Weight(),
            f"PuppiMET_phi_JER_down_{label}": Hist.new.Reg(
                100,
                -3,
                3,
                name=f"PuppiMET_phi_JER_down_{label}",
                label=r"$\phi^{\mathrm{PuppiMET}}$ [GeV]",
            ).Weight(),
            f"W_phi_{label}": Hist.new.Reg(
                60,
                -3.2,
                3.2,
                name=f"W_phi_{label}",
                label=r"$\phi_W$",
            ).Weight(),
            f"W_SUEP_BV_{label}": Hist.new.Reg(
                100,
                -1,
                2,
                name=f"W_SUEP_BV_{label}",
                label="($p_T^W - p_T^{SUEP}$)/$p_T^{SUEP}$",
            ).Weight(),
            f"SUEP_W_BV_{label}": Hist.new.Reg(
                200,
                -1,
                20,
                name=f"W_SUEP_BV_{label}",
                label="($p_T^{SUEP} - p_T^{W}$)/$p_T^{W}$",
            ).Weight(),
            f"deltaPhi_SUEP_W_{label}": Hist.new.Reg(
                60,
                0,
                3.2,
                name=f"deltaPhi_SUEP_W_{label}",
                label=r"$\Delta\phi$(SUEP, W)",
            ).Weight(),
            f"2D_SUEP_pt_vs_W_pt_{label}": Hist.new.Reg(
                100,
                0,
                1000,
                name=f"SUEP_pt_{label}",
                label=r"$p^{\mathrm{SUEP}}_T$ [GeV]",
            )
            .Reg(
                100,
                0,
                1000,
                name=f"W_pt_{label}",
                label=r"$p^{\mathrm{W}}_T$ [GeV]",
            ).Weight(),
    })

    return output

def init_hists_VRGJ(output, label):

    output.update({
        f"sumAK4PhotonMET_pt_{label}": Hist.new.Reg(
            100,
            0,
            1000,
            name=f"sumAK4PhotonMET_pt_{label}",
            label=r"$|\Sigma_{i=0}\vec{p}^{\mathrm{ak4jet,~i}}_T  + \vec{p}^{\mathrm{MET}}_T + \vec{p}^{\gamma}_T|$",
        ).Weight(),
        f"sumAK4PhotonMET_phi_{label}": Hist.new.Reg(
            60,
            -3.2,
            3.2,
            name=f"sumAK4PhotonMET_phi_{label}",
            label=r"$\phi$ of $\Sigma_{i=0}\vec{p}^{\mathrm{ak4jet,~i}}_T  + \vec{p}^{\mathrm{MET}}_T + \vec{p}^{\gamma}_T$",
        ).Weight(),
        f"sumAK4PhotonMET_photon_BV_{label}": Hist.new.Reg(
            100,
            -1,
            2,
            name=f"sumAK4PhotonMET_photon_BV_{label}",
            label=r"($|\Sigma_{i=0}\vec{p}^{\mathrm{ak4jet,~i}}_T  + \vec{p}^{\mathrm{MET}}_T + \vec{p}^{\gamma}_T| - p_T^{\gamma}$)/$p_T^{\gamma}$",
        ).Weight(),
        f"deltaPhi_SUEP_photon_{label}": Hist.new.Reg(
            60,
            0,
            3.2,
            name=f"deltaPhi_SUEP_photon_{label}",
            label=r"$\Delta\phi(\gamma, SUEP)$",
        ).Weight(),
        f"photon_SUEP_BV_{label}": Hist.new.Reg(
            100,
            -1,
            2,
            name=f"photon_SUEP_BV_{label}",
            label=r"($p_T^{\gamma} - p_T^{SUEP}$)/$p_T^{SUEP}$",
        ).Weight(),
        f"SUEP_photon_BV_{label}": Hist.new.Reg(
            100,
            -1,
            20,
            name=f"SUEP_photon_BV_{label}",
            label=r"($p_T^{SUEP} - p_T^{\gamma}$)/$p_T^{\gamma}$",
        ).Weight(),
        f"photon_pt_{label}": Hist.new.Reg(
            200,
            0,
            1000,
            name=f"photon_pt_{label}",
            label=r"$p^{\gamma}_T$",
        ).Weight(),
        f"photon_sieie_{label}": Hist.new.Reg(
            100,
            0,
            10,
            name=f"photon_sieie_{label}",
            label=r"$\sigma^{\gamma}_{i \eta i \eta}$",
        ).Weight(),
        f"photon_phi_{label}": Hist.new.Reg(
            100,
            -3,
            3,
            name=f"photon_phi_{label}",
            label=r"$\phi^{\gamma}$",
        ).Weight(),
        f"photon_eta_{label}": Hist.new.Reg(
            100,
            -3,
            3,
            name=f"photon_eta_{label}",
            label=r"$\eta^{\gamma}$",
        ).Weight(),
        f"photon_pixelSeed_{label}": Hist.new.Reg(
            2,
            0,
            2,
            name=f"photon_pixelSeed_{label}",
            label=r"$\gamma$ pixelSeed",
        ).Weight(),
        f"photon_mvaID_{label}": Hist.new.Reg(
            100,
            -1,
            1,
            name=f"photon_mvaID_{label}",
            label=r"$\gamma$ mvaID",
        ).Weight(),
        f"photon_electronVeto_{label}": Hist.new.Reg(
            2,
            0,
            2,
            name=f"photon_electronVeto_{label}",
            label=r"$\gamma$ electronVeto",
        ).Weight(),
        f"photon_hoe_{label}": Hist.new.Reg(
            100,
            0,
            1,
            name=f"photon_hoe_{label}",
            label=r"$\gamma$ HoE",
        ).Weight(),
        f"photon_r9_{label}": Hist.new.Reg(
            100,
            0,
            1.5,
            name=f"photon_r9_{label}",
            label=r"$\gamma$ r9",
        ).Weight(),
        f"photon_cutBased_{label}": Hist.new.Reg(
            4,
            0,
            4,
            name=f"photon_cutBased_{label}",
            label=r"$\gamma$ cutBased",
        ).Weight(),
        f"photon_pfRelIso03_all_{label}": Hist.new.Reg(
            100,
            0,
            1,
            name=f"photon_pfRelIso03_all_{label}",
            label=r"$\gamma$ pfRelIso03_all",
        ).Weight(),
        f"photon_isScEtaEB_{label}": Hist.new.Reg(
            2,
            0,
            2,
            name=f"photon_isScEtaEB_{label}",
            label=r"$\gamma$ isScEtaEB",
        ).Weight(),
        f"photon_isScEtaEE_{label}": Hist.new.Reg(
            2,
            0,
            2,
            name=f"photon_isScEtaEE_{label}",
            label=r"$\gamma$ isScEtaEE",
        ).Weight(),
        f"minDeltaRJetPhoton_{label}": Hist.new.Reg(
            100,
            0,
            6,
            name=f"minDeltaRJetPhoton_{label}",
            label=r"min($\Delta R$(ak4jets, $\gamma$))",
        ).Weight(),
        f"maxDeltaRJetPhoton_{label}": Hist.new.Reg(
            100,
            0,
            6,
            name=f"maxDeltaRJetPhoton_{label}",
            label=r"max($\Delta R$(ak4jets, $\gamma$))",
        ).Weight(),
        f"minDeltaPhiJetPhoton_{label}": Hist.new.Reg(
            60,
            0,
            3.2,
            name=f"minDeltaPhiJetPhoton_{label}",
            label=r"min($\Delta\phi$(ak4jets, $\gamma$))",
        ).Weight(),
        f"maxDeltaPhiJetPhoton_{label}": Hist.new.Reg(
            60,
            0,
            3.2,
            name=f"maxDeltaPhiJetPhoton_{label}",
            label=r"max($\Delta\phi$(ak4jets, $\gamma$))",
        ).Weight(),
        f"minDeltaEtaJetPhoton_{label}": Hist.new.Reg(
            100,
            0,
            6,
            name=f"minDeltaEtaJetPhoton_{label}",
            label=r"min($\Delta\eta$(ak4jets, $\gamma$))",
        ).Weight(),
        f"maxDeltaEtaJetPhoton_{label}": Hist.new.Reg(
            100,
            0,
            6,
            name=f"maxDeltaEtaJetPhoton_{label}",
            label=r"max($\Delta\eta$(ak4jets, $\gamma$))",
        ).Weight(),
        f"2D_SUEP_pt_vs_photon_pt_{label}": Hist.new.Reg(
            100,
            0,
            1000,
            name=f"SUEP_pt_{label}",
            label="SUEP $p_T$ [GeV]",
        )
        .Reg(
            200,
            0,
            2000,
            name=f"photon_pt_{label}",
            label="Photon $p_T$ [GeV]",
        ).Weight(),
        f"WH_gammaTriggerBits_{label}": Hist.new.Reg(
            20,
            0,
            20,
            name=f"WH_gammaTriggerBits_{label}",
            label=r"WH_gammaTriggerBits",
        ).Weight(),
        f"deltaPhi_photon_MET_{label}": Hist.new.Reg(
            100,
            0,
            6,
            name=f"deltaPhi_photon_MET_{label}",
            label=r"$\Delta \phi$(photon, MET)",
        ).Weight(),
        f"ak4jet1_inSUEPcluster_photon_BV_{label}": Hist.new.Reg(
            100,
            -1,
            2,
            name=f"ak4jet1_inSUEPcluster_photon_BV_{label}",
            label=r"($p_T^{\mathrm{ak4jet1}} - p_T^{\gamma}$)/$p_T^{\gamma}$",
        ).Weight(),
        f"photon_ak4jet1_inSUEPcluster_vBV_{label}": Hist.new.Reg(
            100,
            -1,
            2,
            name=f"photon_ak4jet1_inSUEPcluster_vBV_{label}",
            label=r"|$\vec{p}_T^{\gamma} + \vec{p}_T^{\mathrm{ak4jet1}}$| / $p_T^{\mathrm{ak4jet1}}$",
        ).Weight(),
        f"2D_photon_pt_vs_photon_eta_{label}": Hist.new.Reg(
            200,
            0,
            2000,
            name=f"photon_pt_{label}",
            label="Photon $p_T$ [GeV]",
        )
        .Reg(
            100,
            -3,
            3,
            name=f"photon_eta_{label}",
            label=r"Photon $\eta$",
        )
        .Weight(),
        f"WH_no_doubleCountedPhotons_{label}": Hist.new.Int(
            0,
            2,
            name=f"WH_no_doubleCountedPhotons_{label}",
            label="no double counted photons",
        ).Weight(),
        f"minDeltaRGenRecoPhotons_{label}": Hist.new.Reg(
            100,
            0,
            6,
            name=f"minDeltaRGenRecoPhotons_{label}",
            label=r"min $\Delta R(\mathrm{Gen \gamma, Reco \gamma})$",
        ).Weight(),
        f"looseNotTightLepton1_pt_{label}": Hist.new.Reg(
            1000,
            0,
            1000,
            name=f"looseNotTightLepton_pt_{label}",
            label="Loose not tight lepton $p_T$ [GeV]",
        ).Weight(),
        f"deltaPhi_photon_hardMET_{label}": Hist.new.Reg(
            100,
            0,
            6,
            name=f"deltaPhi_photon_hardMET_{label}",
            label=r"$\Delta \phi$(photon, hard MET)",
        ).Weight(),
        f"3D_SUEP_nconst_vs_WH_gammaTriggerBits_vs_WH_gammaTriggerUnprescaleWeight_{label}": Hist.new.Variable(
            np.linspace(-0.5, 99.5, 101),
            name=f"SUEP_nconst_{label}",
            label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
        )
        .Reg(
            20,
            0,
            20,
            name=f"WH_gammaTriggerBits_{label}",
            label=r"WH_gammaTriggerBits",
        )
        .Variable(
            [0,2,7,9,60,64,245,251,500],
            name=f"WH_gammaTriggerUnprescaleWeight_{label}",
            label=r"WH_gammaTriggerUnprescaleWeight",
        )
        .Weight(),
        f"3D_photon_pt_vs_WH_gammaTriggerBits_vs_WH_gammaTriggerUnprescaleWeight_{label}": Hist.new.Reg(
            100,
            0,
            5000,
            name=f"photon_pt_{label}",
            label="Photon $p_T$ [GeV]",
        )
        .Reg(
            16,
            0,
            16,
            name=f"WH_gammaTriggerBits_{label}",
            label=r"WH_gammaTriggerBits",
        )
        .Variable(
            [0,2,7,9,60,64,245,251,500],
            name=f"WH_gammaTriggerUnprescaleWeight_{label}",
            label=r"WH_gammaTriggerUnprescaleWeight",
        )
        .Weight(),
    })

def init_hists_WHlimits(output, label, regions_list):
    # A minimal set of histograms to be used as inputs to the limits

    output.update({
        f"2D_SUEP_S1_vs_SUEP_nconst_{label}": Hist.new.Reg(
            100,
            0,
            1.0,
            name=f"SUEP_S1_{label}",
            label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
        )
        .Variable(
            np.linspace(-0.5, 199.5, 201),
            name=f"nconst_{label}",
            label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
        )
        .Weight(),
    })

    for r in regions_list:

        # these histograms will be made for each systematic, and each ABCD region
        output.update(
            {
                f"{r}SUEP_nconst_{label}": Hist.new.Variable(
                    np.linspace(-0.5, 299.5, 301),
                    name=f"{r}SUEP_nconst_{label}",
                    label=r"$n^{\mathrm{SUEP}}_{\mathrm{constituent}}$",
                ).Weight(),
                f"{r}SUEP_S1_{label}": Hist.new.Reg(
                    100,
                    0,
                    1,
                    name=f"{r}SUEP_S1_{label}",
                    label=r"$S^{\mathrm{SUEP}}_{\mathrm{boosted}}$",
                ).Weight()
            }
        )
