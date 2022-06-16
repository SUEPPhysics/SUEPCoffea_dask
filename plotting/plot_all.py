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
parser.add_argument('--weights', type=str, default='None', help="Pass the filename of the weights, e.g. --weights weights.npy")
parser.add_argument('--xrootd', type=int, default=0, help="Local data or xrdcp from hadoop (default=False)")
options = parser.parse_args()

def call_makeplots(cmd):
    """ This runs in a separate thread. """
    print("----[%] :", cmd)
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)

pool = ThreadPool(multiprocessing.cpu_count())

def read_filelist(file):
    with open(file,'r') as f:
        lines = f.readlines()
        lines = [l.split('/')[-1].strip('\n') for l in lines]
    return lines

QCD_HT_2018_scout = read_filelist('../filelist/list_2018_scout_MC.txt')
SUEP = [f for f in read_filelist('../filelist/list_2018_MC_A01.txt') if 'SUEP' in f]
QCD_HT_2018 = [f for f in read_filelist('../filelist/list_2018_MC_A01.txt') if 'QCD_HT' in f]
QCD_HT_2017 = [f for f in read_filelist('../filelist/list_2017_MC_A01.txt') if 'QCD_HT' in f]
QCD_HT_2016 = [f for f in read_filelist('../filelist/list_2016_MC_A01.txt') if 'QCD_HT' in f]
QCD_Pt_2018 = [f for f in read_filelist('../filelist/list_2018_MC_A01.txt') if 'QCD_Pt' in f]
QCD_Pt_2017 = [f for f in read_filelist('../filelist/list_2017_MC_A01.txt') if 'QCD_Pt' in f]
QCD_Pt_2016 = [f for f in read_filelist('../filelist/list_2016_MC_A01.txt') if 'QCD_Pt' in f]
data_2018 = read_filelist('../filelist/list_2018_data_A01.txt')
data_2017 = read_filelist('../filelist/list_2017_data_A01.txt')
data_2016 = read_filelist('../filelist/list_2016_data_A01.txt')

results = []
start = time.time()

# for sample in data_2018:
#     cmd = 'python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --weights={} --isMC=0 --era={}'.format(options.tag, options.output, sample, options.xrootd, options.weights, 2018)
#     results.append(pool.apply_async(call_makeplots, (cmd,))) 
#for sample in QCD_2018:
#    cmd = 'python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --isMC=1 --era={}'.format(options.tag, options.output, sample, options.xrootd, 2018)
#    results.append(pool.apply_async(call_makeplots, (cmd,))) 
# for sample in QCD_HT_2018:
#     cmd = 'python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --weights={} --isMC=1 --era={}'.format(options.tag, options.output, sample, options.xrootd, options.weights, 2018)
#     results.append(pool.apply_async(call_makeplots, (cmd,))) 
for sample in QCD_HT_2018_scout:
    cmd = 'python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --weights={} --isMC=1 --era={} --scouting={} '.format(options.tag, options.output, sample, options.xrootd, options.weights, 2018, 1)
    results.append(pool.apply_async(call_makeplots, (cmd,))) 
# for sample in QCD_Pt_2017:
#      cmd = 'python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --isMC=1 --era={}'.format(options.tag, options.output, sample, options.xrootd, 2017)
#      results.append(pool.apply_async(call_makeplots, (cmd,)))
# for sample in QCD_HT_2017:
#      cmd = 'python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --isMC=1 --era={}'.format(options.tag, options.output, sample, options.xrootd, 2017)
#      results.append(pool.apply_async(call_makeplots, (cmd,)))
#for sample in QCD_HT_2016:
#     cmd = 'python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --isMC=1 --era={}'.format(options.tag, options.output, sample, options.xrootd, 2016)
#     results.append(pool.apply_async(call_makeplots, (cmd,)))
#for sample in QCD_Pt_2016:
#     cmd = 'python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --isMC=1 --era={}'.format(options.tag, options.output, sample, options.xrootd, 2016)
#     results.append(pool.apply_async(call_makeplots, (cmd,)))
# for sample in data_2017:
#      cmd = 'python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --isMC=0 --era={}'.format(options.tag, options.output, sample, options.xrootd, 2017)
#      results.append(pool.apply_async(call_makeplots, (cmd,))) 
#for sample in data_2016:
#     cmd = 'python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --isMC=0 --era={}'.format(options.tag, options.output, sample, options.xrootd, 2016)
#     results.append(pool.apply_async(call_makeplots, (cmd,))) 
# for sample in SUEP:
#     cmd = 'python3 make_plots.py --tag={} --output={} --dataset={} --xrootd={} --weights={} --isMC=1'.format(options.tag, options.output, sample, options.xrootd, options.weights)
#     results.append(pool.apply_async(call_makeplots, (cmd,))) 

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
