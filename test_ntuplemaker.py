import os
import sys
import subprocess
from time import time

runs = {
    # 'ggF-Offline': {
    #     'script': 'condor_SUEP_WS.py',
    #     'options': [
    #         '--isMC 1 --era 2018 --doSyst 1',
    #         '--isMC 0 --era 2018 --doSyst 1',
    #     ],
    #     'root_files': [
    #         'root://xrootd.cmsaf.mit.edu//store/user/paus/nanosu/A02/QCD_Pt_800to1000_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM//FFAB73E1-5B63-0B43-867E-B35DB3C75E60.root',
    #         'root://xrootd.cmsaf.mit.edu//store/user/paus/nanosu/A02/JetHT+Run2018C-UL2018_MiniAODv2-v1+MINIAOD//FFE35F7A-1786-BE4F-8AF4-66DEA58012F3.root'
    #     ],
    #     'out_file': 'out.hdf5'
    # },

    'ggF-Scouting': {
        'script': 'condor_Scouting.py',
        'options': [
            '--isMC 1 --era 2018 --doSyst 1',
            '--isMC 0 --era 2018 --doSyst 1',
        ],
        'root_files': [
            'root://xrootd.cmsaf.mit.edu//store/user/paus/nanosc/E07/QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18RECO-106X_upgrade2018_realistic_v11_L1v1-v2+AODSIM//FFEBDF0B-33D5-F84A-B899-0F8EF89FA734.root',
            'root://xrootd.cmsaf.mit.edu//store/user/paus/nanosc/E08/ScoutingPFCommissioning+Run2016B-v2+RAW//FAF17B72-201D-E611-BA11-02163E012571.root'
        ],
        'out_file': 'out.hdf5'
    },
    
    'WH': {
        'script': 'condor_SUEP_WH.py',
        'options': [
            '--isMC 1 --era 2018 --doSyst 1',
            '--isMC 0 --era 2018 --doSyst 1',
        ],
        'root_files': [
            'root://xrootd.cmsaf.mit.edu//store/user/paus/nanosu/A02/QCD_Pt_800to1000_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM//FFAB73E1-5B63-0B43-867E-B35DB3C75E60.root',
            'root://xrootd.cmsaf.mit.edu//store/user/paus/nanosu/A02/JetHT+Run2018C-UL2018_MiniAODv2-v1+MINIAOD//FFE35F7A-1786-BE4F-8AF4-66DEA58012F3.root'
        ],
        'out_file': 'out.hdf5'
    }
}

def execute_test(run, config):

    script = config['script']
    options = config['options']
    root_files = config['root_files']
    output_file = config['out_file']

    for option, root_file in zip(options, root_files):
        command = f'python {script} --infile {root_file} {option}'
        print("Executing command:", command)
        subprocess.run(command, shell=True)
        
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f'PASSED! Output file for run {run} exists and is not empty.')
        else:
            sys.exit(f'FAILED! Output file for run {run} does not exist or is empty.')
        os.remove(output_file)

    print("All tests passed for run", run)

startTot = time()
for run, config in runs.items():
    start = time()
    print("Running test for run", run, "...")
    execute_test(run, config)
    end = time()
    print(f"Time to take the {run} test:", end - start, "seconds")
endTot = time()
print()
print("Total time to take the tests:", endTot - startTot, "seconds")