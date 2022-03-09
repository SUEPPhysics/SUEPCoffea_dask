import os
import subprocess
import shlex
import argparse
import multiprocessing
from multiprocessing.pool import ThreadPool
import time

parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan"  , help="Production tag", required=True)
parser.add_argument("-o"   , "--output"   , type=str, default="IronMan"  , help="Output tag", required=False)
parser.add_argument('--xrootd', type=int, default=0, help="Local data or xrdcp from hadoop (default=False)")
options = parser.parse_args()

def call_makeplots(cmd):
    """ This runs in a separate thread. """
    print("----[%] :", cmd)
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)

pool = ThreadPool(multiprocessing.cpu_count())

SUEP = [
           "SUEP-m1000-darkPho+RunIIAutumn18-private+MINIAODSIM",
           "SUEP-m1000-darkPhoHad+RunIIAutumn18-private+MINIAODSIM",
           "SUEP-m1000-generic+RunIIAutumn18-private+MINIAODSIM",
           "SUEP-m125-darkPho+RunIIAutumn18-private+MINIAODSIM",
           "SUEP-m125-darkPhoHad+RunIIAutumn18-private+MINIAODSIM",
           "SUEP-m125-generic+RunIIAutumn18-private+MINIAODSIM",
           "SUEP-m400-darkPho+RunIIAutumn18-private+MINIAODSIM",
           "SUEP-m400-darkPhoHad+RunIIAutumn18-private+MINIAODSIM",
           "SUEP-m400-generic+RunIIAutumn18-private+MINIAODSIM",
           "SUEP-m750-darkPho+RunIIAutumn18-private+MINIAODSIM",
           "SUEP-m750-darkPhoHad+RunIIAutumn18-private+MINIAODSIM",
           "SUEP-m750-generic+RunIIAutumn18-private+MINIAODSIM"
]

QCD = [ 
           "QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_120to170_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2+MINIAODSIM",
           "QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_15to30_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_170to300_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_300to470_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_30to50_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-pilot_106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_470to600_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_50to80_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_600to800_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_800to1000_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
           "QCD_Pt_80to120_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM",
]

data = [
           "JetHT+Run2018A-UL2018_MiniAODv2-v1+MINIAOD",
           "JetHT+Run2018B-UL2018_MiniAODv2-v1+MINIAOD",
           "JetHT+Run2018C-UL2018_MiniAODv2-v1+MINIAOD",
           "JetHT+Run2018D-UL2018_MiniAODv2-v1+MINIAOD"
]

results = []
start = time.time()

for sample in QCD:
    cmd = 'python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --isMC=1'.format(options.tag, options.output, sample, options.xrootd)
    results.append(pool.apply_async(call_makeplots, (cmd,)))
for sample in SUEP:
    cmd = 'python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --isMC=1'.format(options.tag, options.output, sample, options.xrootd)
    results.append(pool.apply_async(call_makeplots, (cmd,))) 
for sample in data:
    cmd = 'python3 make_plots.py --tag={} --output={} -dataset={} --xrootd={} --isMC=0'.format(options.tag, options.output, sample, options.xrootd)
    results.append(pool.apply_async(call_makeplots, (cmd,)))

# Close the pool and wait for each running task to complete
pool.close()
pool.join() 
for result in results:
    out, err = result.get()
    if "No such file or directory" in str(err):
        print(str(err))
        print(" ----------------- ")
        print() 
end = time.time()
print("All done! plot_all.py took",round(end - start),"seconds to run.")
