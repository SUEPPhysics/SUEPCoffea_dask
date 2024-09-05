"""
Script to run the SUEPDaskHistMaker for the WH analysis.
Author: Luca Lavezzo
Date: August 2024
"""

import os
import argparse
from types import SimpleNamespace

import fill_utils
import hist_defs
import var_defs
from SUEPDaskHistMaker import SUEPDaskHistMaker


def getOptions() -> dict:

    username = os.environ["USER"]

    parser = argparse.ArgumentParser(description='Run the SUEP histogram maker')

    # arguments for this script
    parser.add_argument('--input', '-i', type=str, help='Text file of samples to process')
    parser.add_argument('--sample', type=str, help='Sample to process, alternative to --input')
    parser.add_argument('--client', type=str, default='local', choices=['local', 'slurm'], help='Where to set up the client')
    parser.add_argument('--nworkers', '-n', type=int, default=20, help='Number of workers to use')

    # reuired SUEPDaskHistMaker arguments
    parser.add_argument('--isMC', type=int, default=1, help='isMC')
    parser.add_argument('--channel', type=str, default='WH', help='Channel')
    parser.add_argument('--era', type=str, required=True, help='Era')
    parser.add_argument('--tag', type=str, required=True, help='Ntuple tag')
    parser.add_argument('--output', type=str, required=True, help='Output tag')

    # optional SUEPDaskHistMaker arguments
    parser.add_argument('--doSyst', type=int, default=0, help='Do systematics')
    parser.add_argument('--verbose', type=int, default=0, help='Verbose')
    parser.add_argument('--xrootd', type=int, default=0, help='Use xrootd to read the ntuples')
    parser.add_argument('--saveDir', type=str, default='/ceph/submit/data/user/'+username[0]+'/'+username+'/SUEP/outputs/', help='Save directory')
    parser.add_argument('--logDir', type=str, default='/work/submit/'+username+'/SUEP/logs/', help='Log directory')
    parser.add_argument('--dataDirLocal', type=str, default='/ceph/submit/data/user/'+username[0]+'/'+username+'/SUEP/{}/{}/', help='Local ntuple directory')
    parser.add_argument('--dataDirXRootD', type=str, default='/cms/store/user/'+username+'/SUEP/{}/{}/', help='XRootD ntuple directory')
    parser.add_argument('--merged', type=int, default=0, help='Use merged ntuples')
    parser.add_argument('--maxFiles', type=int, default=-1, help='Maximum number of files to run on')
    parser.add_argument('--pkl', type=int, default=1, help='Use pickle files')
    parser.add_argument('--printEvents', type=int, default=0, help='Print events')
    parser.add_argument('--redirector', type=str, default='root://submit50.mit.edu/', help='Redirector')
    parser.add_argument('--doABCD', type=int, default=0, help='Do ABCD')
    parser.add_argument('--file', type=str, help="Ntuple single filename, in case you don't want to process a whole sample")

    options = parser.parse_args()
    options = vars(options)

    # either input or sample must be provided
    if not options.get('input') and not options.get('sample'):
        raise ValueError('Either --input or --sample must be provided')
    if options.get('input'):
        with open(options['input']) as f:
            options['samples'] = [line.split('/')[-1] for line in f.read().splitlines()]
    else:
        options['samples'] = [options['sample']]

    return options


def main():

    # get the options
    options = getOptions()
   
    # set up your configuration
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
                #"SUEP_S1_HighestPT < 0.3",
                #"SUEP_nconst_HighestPT < 40",
            ],
            "syst":  [],
        },
        # "CRWJmu": {
        #     "input_method": "HighestPT",
        #     "method_var": "SUEP_nconst_HighestPT",
        #     "SR": [
        #         ["SUEP_S1_HighestPT", ">=", 0.3],
        #         ["SUEP_nconst_HighestPT", ">=", 40],
        #     ],
        #     "selections": [
        #         "WH_MET_pt > 30",
        #         "W_pt > 40",
        #         "W_mt < 130",
        #         "W_mt > 30",
        #         "bjetSel == 1",
        #         "deltaPhi_SUEP_W > 1.5",
        #         "deltaPhi_SUEP_MET > 1.5",
        #         "deltaPhi_lepton_SUEP > 1.5",
        #         "ak4jets_inSUEPcluster_n_HighestPT >= 1",
        #         "W_SUEP_BV < 2",
        #         "deltaPhi_minDeltaPhiMETJet_MET > 1.5",
        #         "SUEP_S1_HighestPT < 0.3",
        #         "SUEP_nconst_HighestPT < 40",
        #         "isMuon == 1",
        #     ],
        #     "syst":  [],
        # },
        # "CRWJe": {
        #     "input_method": "HighestPT",
        #     "method_var": "SUEP_nconst_HighestPT",
        #     "SR": [
        #         ["SUEP_S1_HighestPT", ">=", 0.3],
        #         ["SUEP_nconst_HighestPT", ">=", 40],
        #     ],
        #     "selections": [
        #         "WH_MET_pt > 30",
        #         "W_pt > 40",
        #         "W_mt < 130",
        #         "W_mt > 30",
        #         "bjetSel == 1",
        #         "deltaPhi_SUEP_W > 1.5",
        #         "deltaPhi_SUEP_MET > 1.5",
        #         "deltaPhi_lepton_SUEP > 1.5",
        #         "ak4jets_inSUEPcluster_n_HighestPT >= 1",
        #         "W_SUEP_BV < 2",
        #         "deltaPhi_minDeltaPhiMETJet_MET > 1.5",
        #         "SUEP_S1_HighestPT < 0.3",
        #         "SUEP_nconst_HighestPT < 40",
        #         "isElectron == 1",
        #     ],
        #     "syst":  [],
        # },
        # "VRGJ": {
        #     "input_method": "HighestPT",
        #     "method_var": "SUEP_nconst_HighestPT",
        #     "SR": [
        #         ["SUEP_S1_HighestPT", ">=", 0.3],
        #         ["SUEP_nconst_HighestPT", ">=", 40],
        #     ],
        #     "selections": [
        #         "gammaTriggerSel == 1",
        #         "photon_pt > 40",
        #         "bjetSel == 1",
        #         "deltaPhi_SUEP_photon > 1.5",
        #         "ak4jets_inSUEPcluster_n_HighestPT >= 1",
        #         "photon_SUEP_BV < 2",
        #         #"SUEP_S1_HighestPT < 0.3",
        #         #"SUEP_nconst_HighestPT < 40",
        #     ],
        # }
    }
    hists = {}
    for output_method in config.keys():
        var_defs.initialize_new_variables(output_method, SimpleNamespace(**options), config[output_method])
        if options.get("doSyst", False):
            for syst in config[output_method].get("syst", []):
                if any([j in syst for j in ["JER", "JES"]]):
                    config = fill_utils.get_jet_correction_config(config, syst)
    if options.get("doSyst", False): config = fill_utils.get_track_killing_config(config)
    
    # run the SUEP histogram maker
    histmaker = SUEPDaskHistMaker(config=config, options=options, hists=hists)
    if options['client'] == 'local':
        client = histmaker.setupLocalClient(options['nworkers'])
    else:
        client = histmaker.setupSlurmClient(n_workers=options['nworkers'], min_workers=2, max_workers=200)
    histmaker.run(client, options['samples'])
    client.close()


if __name__ == "__main__":
    main()