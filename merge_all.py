import os
import subprocess
import shlex
import argparse
import multiprocessing
from multiprocessing.pool import ThreadPool
import time

parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan"  , help="Production tag", required=True)
options = parser.parse_args()

print("""WARNING: This wrapper will launch a process per dataset with this tag, 
        each of which use large amounts of memory, depending on how many events
        are kept in each merged file, so make sure to run it somewhere where
        you have enough memory available.""")

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

QCD_HT_2018_scout = read_filelist('filelist/list_2018_scout_MC.txt')
data_2018_scout = read_filelist('filelist/list_2018_scout_data.txt')
SUEP = [f for f in read_filelist('filelist/list_2018_MC_A01.txt') if 'SUEP' in f]
QCD_HT_2018 = [f for f in read_filelist('filelist/list_2018_MC_A01.txt') if 'QCD_HT' in f]
QCD_HT_2017 = [f for f in read_filelist('filelist/list_2017_MC_A01.txt') if 'QCD_HT' in f]
QCD_HT_2016 = [f for f in read_filelist('filelist/list_2016_MC_A01.txt') if 'QCD_HT' in f]
QCD_Pt_2018 = [f for f in read_filelist('filelist/list_2018_MC_A01.txt') if 'QCD_Pt' in f]
QCD_Pt_2017 = [f for f in read_filelist('filelist/list_2017_MC_A01.txt') if 'QCD_Pt' in f]
QCD_Pt_2016 = [f for f in read_filelist('filelist/list_2016_MC_A01.txt') if 'QCD_Pt' in f]
data_2018 = read_filelist('filelist/list_2018_data_A01.txt')
data_2017 = read_filelist('filelist/list_2017_data_A01.txt')
data_2016 = read_filelist('filelist/list_2016_data_A01.txt')

results = []
start = time.time()

# for sample in data_2018:
#     cmd = 'python3 merge_plots.py --tag={} --dataset={} --isMC={}'.format(options.tag, sample, 0)
#     results.append(pool.apply_async(call_makeplots, (cmd,))) 
for sample in data_2018_scout:
    cmd = 'python3 merge_plots.py --tag={} --dataset={} --isMC={}'.format(options.tag, sample, 0)
    results.append(pool.apply_async(call_makeplots, (cmd,))) 
# for sample in QCD_2018:
#     cmd = 'python3 merge_plots.py --tag={} --dataset={} --isMC={}'.format(options.tag, sample, 1)
#     results.append(pool.apply_async(call_makeplots, (cmd,))) 
# for sample in QCD_HT_2018:
#     cmd = 'python3 merge_plots.py --tag={} --dataset={} --isMC={}'.format(options.tag, sample, 1)
#     results.append(pool.apply_async(call_makeplots, (cmd,)))
# for sample in QCD_HT_2018_scout:
#     cmd = 'python3 merge_plots.py --tag={} --dataset={} --isMC={}'.format(options.tag, sample, 1)
#     results.append(pool.apply_async(call_makeplots, (cmd,))) 
# for sample in QCD_HT_2017:
#     cmd = 'python3 merge_plots.py --tag={} --dataset={} --isMC={}'.format(options.tag, sample, 1)
#     results.append(pool.apply_async(call_makeplots, (cmd,)))
# for sample in QCD_2017:
#     cmd = 'python3 merge_plots.py --tag={} --dataset={} --isMC={}'.format(options.tag, sample, 1)
#     results.append(pool.apply_async(call_makeplots, (cmd,)))
# for sample in QCD_HT_2016:
#     cmd = 'python3 merge_plots.py --tag={} --dataset={} --isMC={}'.format(options.tag, sample)
#     results.append(pool.apply_async(call_makeplots, (cmd,)))
# for sample in data_2017:
#     cmd = 'python3 merge_plots.py --tag={} --dataset={} --isMC={}'.format(options.tag, sample)
#     results.append(pool.apply_async(call_makeplots, (cmd,))) 
# for sample in data_2016:
#     cmd = 'python3 merge_plots.py --tag={} --dataset={} --isMC={}'.format(options.tag, sample)
#     results.append(pool.apply_async(call_makeplots, (cmd,))) 
# for sample in SUEP:
#     cmd = 'python3 merge_plots.py --tag={} --dataset={} --isMC={}'.format(options.tag, sample, 1)
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
print("All done! merge_all.py took",round(end - start),"seconds to run.")
