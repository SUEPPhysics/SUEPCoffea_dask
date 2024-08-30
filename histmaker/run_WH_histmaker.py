import os
import argparse
from types import SimpleNamespace, Namespace

import fill_utils
import hist_defs
import var_defs
from SUEPDaskHistMaker import SUEPDaskHistMaker


def getOptions() -> Namespace:

    username = os.environ["USER"]

    parser = argparse.ArgumentParser(description='Run the SUEP histogram maker')

    # arguments for this script
    parser.add_argument('--input', '-i', type=str, required=True, help='Text file of samples to process')
    parser.add_argument('--client', type=str, default='local', choices=['local', 'dask'], help='Where to set up the client')

    # reuired SUEPDaskHistMaker arguments
    parser.add_argument('--isMC', type=int, default=1, help='isMC')
    parser.add_argument('--channel', type=str, default='WH' help='Channel')
    parser.add_argument('--era', type=str, required=True, help='Era')
    parser.add_argument('--tag', type=str, required=True, help='Ntuple tag')
    parser.add_argument('--output', type=str, required=True, help='Output tag')

    # optional SUEPDaskHistMaker arguments
    parser.add_argument('--doSyst', type=int, default=0, help='Do systematics')
    parser.add_argument('--verbose', type=int, default=0, help='Verbose')
    parser.add_argument('--xrootd', type=int, default=0, help='Use xrootd to read the ntuples')
    parser.add_argument('--saveDir', type=str, default='/ceph/submit/data/user/'+username[0]+'/'+username+'/SUEP/outputs/', help='Save directory')
    parser.add_argument('--logDir', type=str, default='/work/submit/'+username+'/SUEP/logs/', help='Log directory')
    parser.add_argument('--dataDirLocal', type=str, default='/data/submit/cms/store/user/'+username+'/SUEP/{}/{}/', help='Local ntuple directory')
    parser.add_argument('--dataDirXRootD', type=str, default='/cms/store/user/'+username+'/SUEP/{}/{}/', help='XRootD ntuple directory')
    parser.add_argument('--merged', type=int, default=0, help='Use merged ntuples')
    parser.add_argument('--maxFiles', type=int, default=-1, help='Maximum number of files to run on')
    parser.add_argument('--pkl', type=int, default=1, help='Use pickle files')
    parser.add_argument('--printEvents', type=int, default=0, help='Print events')
    parser.add_argument('--scouting', type=int, default=0, help='Scouting')
    parser.add_argument('--redirector', type=str, default='root://submit50.mit.edu/', help='Redirector')

    options = parser.parse_args()
    return dict(options)


def main():

    options = getOptions()

    with open(options['input']) as f:
        samples = f.read().splitlines()

    config = {
        "CRWJ": {
            "input_method": "HighestPT",
            "method_var": "SUEP_nconst_HighestPT",
            "SR": [
                ["SUEP_S1_HighestPT", ">=", 0.3],
                ["SUEP_nconst_HighestPT", ">=", 40],
            ],
            "selections": [
                "WH_MET_pt > 30",
                "W_pt > 40",
                "W_mt < 130",
                "W_mt > 30",
                "bjetSel == 1",
                "deltaPhi_SUEP_W > 1.5",
                "deltaPhi_SUEP_MET > 1.5",
                "deltaPhi_lepton_SUEP > 1.5",
                "ak4jets_inSUEPcluster_n_HighestPT >= 1",
                "W_SUEP_BV < 2",
                "deltaPhi_minDeltaPhiMETJet_MET > 1.5",
                "SUEP_S1_HighestPT < 0.3",
                "SUEP_nconst_HighestPT < 40",
            ],
            "syst":  [
                "puweights_up",
                "puweights_down"
            ],
        },
    }
    hists = {}
    for output_method in config.keys():
        var_defs.initialize_new_variables(output_method, SimpleNamespace(**options), config[output_method])
        hist_defs.initialize_histograms(hists, output_method, SimpleNamespace(**options), config[output_method])
        if options.get("doSyst", False):
            for syst in config[output_method].get("syst", []):
                if any([j in syst for j in ["JER", "JES"]]):
                    config = fill_utils.get_jet_correction_config(config, syst)
    if options.get("doSyst", False): config = fill_utils.get_track_killing_config(config)
    
    
    histmaker = SUEPDaskHistMaker(config=config, options=options, hists=hists)
    if options['client'] == 'local':
        client = histmaker.setupLocalClient(10)
    else:
        client = histmaker.setupSlurmClient(n_workers=100, min_workers=2, max_workers=200)
    histmaker.run(client, samples)
    client.close()


if __name__ == "__main__":
    main()