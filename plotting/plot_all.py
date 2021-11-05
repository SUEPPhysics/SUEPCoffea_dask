import os
import subprocess
import shlex
import argparse
import multiprocessing
from multiprocessing.pool import ThreadPool
import time

parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan"  , help="Production tag", required=True)
dhelp = """
Whic dataset(s) to plot. Currently supported:
MC (all QCD, SUEP),
MC_QCD (each QCD HT bin separately)
"""
parser.add_argument("-d"   , "--dataset"   , type=str, default="MC"  , help=dhelp, required=False)
options = parser.parse_args()


def call_makeplots(cmd):
    """ This runs in a separate thread. """
    print("----[%] :", cmd)
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)

pool = ThreadPool(multiprocessing.cpu_count())



datasets_MC = [
           "QCD",
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

datasets_MC_QCD = [
           "QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8+RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1+MINIAODSIM",
           "QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8+RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1+MINIAODSIM",
           "QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8+RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1+MINIAODSIM",
           "QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8+RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1+MINIAODSIM",
           "QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8+RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1+MINIAODSIM",
           "QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8+RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1+MINIAODSIM",
           "QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8+RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1+MINIAODSIM",
           "QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8+RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1+MINIAODSIM",    
]

datas = [
           "JetHT+Run2018A-17Sep2018-v1+MINIAOD",
           "JetHT+Run2018B-17Sep2018-v1+MINIAOD",
           "JetHT+Run2018C-17Sep2018-v1+MINIAOD",
           "JetHT+Run2018D-PromptReco-v1+MINIAOD",
           "JetHT+Run2018D-PromptReco-v2+MINIAOD"
         ]

results = []
start = time.time()
if options.dataset == 'MC':
    for dataset in datasets_MC:
         cmd = 'python3 make_plots.py --isMC=1 --tag={} --dataset={}'.format(options.tag, dataset)
         results.append(pool.apply_async(call_makeplots, (cmd,)))
if options.dataset == 'MC_QCD':
    for dataset in datasets_MC_QCD:
         cmd = 'python3 make_plots.py --isMC=0 --tag={} --dataset={}'.format(options.tag, dataset)
         results.append(pool.apply_async(call_makeplots, (cmd,)))
if options.dataset == 'data':
    for dataset in datas:
         cmd = 'python3 make_plots.py --isMC=1 --tag={} --dataset={}'.format(options.tag, dataset)
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
