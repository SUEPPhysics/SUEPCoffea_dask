import glob
import plot_utils
import fill_utils

def loader(tag="test", load_data=False, scale_qcd=0, verbosity=0):
    # input .pkl files
    plotDir = f'./{tag}_output_histograms/'
    infile_names = glob.glob(plotDir+'*.pkl')

    # generate list of files that you want to merge histograms for
    offline_files_SUEP = [
        f for f in infile_names if ("SUEP" in f) and ("histograms.pkl" in f)
    ]
    offline_files_normalized = [
        f for f in infile_names if ("normalized.pkl" in f)
    ]
    offline_files_other = [
        f for f in infile_names if ("pythia8" in f) and ("histograms.pkl" in f) and ("SUEP" not in f)
    ]
    offline_files = offline_files_normalized + offline_files_other
    if verbosity > 0:
        print(offline_files)

    other_bkg_names = {
        "DY0JetsToLL": "DY0JetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_NLO": "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYLowMass_NLO": "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8+"
                        "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
        "DYLowMass_LO": "DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8+"
                        "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
        "TTJets": "TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8+"
                "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
        "TTToHadronic": "TTToHadronic_TuneCP5_13TeV-powheg-pythia8+"
                "RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
        "TTToSemiLeptonic": "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8+"
                "RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
        "TTTo2L2Nu": "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8+"
                "RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
        "ttZJets": "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8+"
                "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
        "WWZ_4F": "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8+"
                "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v2+MINIAODSIM",
        "ZZTo4L": "ZZTo4L_TuneCP5_13TeV_powheg_pythia8+"
                "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
        "ZZZ": "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8+"
            "RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v2+MINIAODSIM",
        "WJets_inclusive": "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8+"
                    "RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
        "ST_tW": "ST_tW_Dilept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8+"
                "RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM"
    }

    # merge the histograms, apply lumis, exclude low HT bins
    plots_SUEP_2018 = plot_utils.loader(offline_files_SUEP, year=2018)
    plots_2018 = plot_utils.loader(offline_files, year=2018)
    if load_data:
        plots_data = plot_utils.loader(data_files, year=2018, is_data=True)
    
    if verbosity > 1:
        print(plots_SUEP_2018)

    # put everything in one dictionary, apply lumi for SUEPs
    plots = {}
    for key in plots_SUEP_2018.keys():
        tag = ""
        if "SUEP-m" in key:
            tag = "+RunIIAutumn18-private+MINIAODSIM"
        plots[key + "_2018"] = fill_utils.apply_normalization(
            plots_SUEP_2018[key],
            fill_utils.getXSection(
                key + tag, "2018", SUEP=True
            ),
        )
    for key in plots_2018.keys():
        is_binned = False
        binned_samples = [
            "QCD_Pt",
            "WJetsToLNu_HT",
            "WZTo",
            "WZ_all",
            "WWTo",
            "WW_all",
            "ST_t-channel",
            "JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
            "DYNJetsToLL",
        ]
        for binned_sample in binned_samples:
            if binned_sample in key:
                is_binned = True
        if is_binned and ("normalized" not in key) and ("cutflow" in key):
            continue
        if is_binned or ("bkg" in key):
            plots[key + "_2018"] = plots_2018[key]
        else:
            plots[key + "_2018"] = fill_utils.apply_normalization(
                plots_2018[key],
                fill_utils.getXSection(other_bkg_names[key], "2018", SUEP=False),
            )

    # # Combine DYJetsToLL_NLO with DYLowMass_NLO
    # dy_nlo_all = {}
    # for plt_i in plots["DYLowMass_NLO_2018"].keys():
    #     dy_nlo_all[plt_i] = (
    #         plots["DYLowMass_NLO_2018"][plt_i] + plots["DYJetsToLL_NLO_2018"][plt_i]
    #     )
    # plots["DY_2018"] = dy_nlo_all

    # # Combine TTbar powheg
    # ttbar_powheg = {}
    # for plt_i in plots["TTToHadronic_2018"].keys():
    #     ttbar_powheg[plt_i] = (
    #         plots["TTToHadronic_2018"][plt_i] 
    #         + plots["TTToSemiLeptonic_2018"][plt_i]
    #         + plots["TTTo2L2Nu_2018"][plt_i]
    #     )
    # plots["TT_powheg_2018"] = ttbar_powheg

    # # Combine ZZZ with WWZ
    # vvv_combined = {}
    # for plt_i in plots["WWZ_4F_2018"].keys():
    #     vvv_combined[plt_i] = plots["WWZ_4F_2018"][plt_i] + plots["ZZZ_2018"][plt_i]
    # plots["VVV_2018"] = vvv_combined

    # # Combine ZZ, WZ, and WW
    # vv_combined = {}
    # for plt_i in plots["WZ_all_2018"].keys():
    #     vv_combined[plt_i] = (
    #         plots["WW_all_2018"][plt_i]
    #         + plots["WZ_all_2018"][plt_i]
    #         + plots["ZZTo4L_2018"][plt_i]
    #     )
    # plots["VV_2018"] = vv_combined

    # # Combine ST
    # st_combined = {}
    # for plt_i in plots["ST_t-channel_2018"].keys():
    #     st_combined[plt_i] = (
    #         plots["ST_t-channel_2018"][plt_i] + plots["ST_tW_2018"][plt_i]
    #     )
    # plots["ST_2018"] = st_combined

    # # Combine WJetsHT and WJets_inclusive
    # wjets_combined = {}
    # for plt_i in plots["WJetsToLNu_HT_2018"].keys():
    #     wjets_combined[plt_i] = (
    #         plots["WJetsToLNu_HT_2018"][plt_i] + plots["WJets_inclusive_2018"][plt_i]
    #     )
    # plots["WJets_all_2018"] = wjets_combined

    # others_combined = {}
    # for plt_i in plots["ST_t-channel_2018"].keys():
    #     others_combined[plt_i] = (
    #         plots["VVV_2018"][plt_i]
    #         + plots["ST_2018"][plt_i]
    #         + plots["WJets_all_2018"][plt_i]
    #         + plots["ttZJets_2018"][plt_i]
    #     )
    # plots["Other_2018"] = others_combined

    # # Normalize QCD MuEnriched if it exists
    # if "QCD_Pt_MuEnriched_2018" in plots.keys() and scale_qcd > 0:
    #     for plot in plots["QCD_Pt_MuEnriched_2018"].keys():
    #         plots["QCD_Pt_MuEnriched_2018"][plot] = plots["QCD_Pt_MuEnriched_2018"][
    #             plot
    #         ]
    # plots["QCD_2018"] = plots["QCD_Pt_MuEnriched_2018"]

    return plots
