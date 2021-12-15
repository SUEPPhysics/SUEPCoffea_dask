import os
import subprocess
import shlex
import argparse
import multiprocessing
from multiprocessing.pool import ThreadPool
import time

parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan"  , help="Production tag", required=True)
parser.add_argument("-c"   , "--combined"   , dest='combined',  default=False, action='store_true', help="run data samples combined or not", required=False)
options = parser.parse_args()


def call_makeplots(cmd):
    """ This runs in a separate thread. """
    print("----[%] :", cmd)
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)

pool = ThreadPool(multiprocessing.cpu_count())

data_samples = [
           "JetHT+Run2018A-UL2018_MiniAODv2-v1+MINIAOD",
           "JetHT+Run2018C-UL2018_MiniAODv2-v1+MINIAOD",
           "JetHT+Run2018B-UL2018_MiniAODv2-v1+MINIAOD",
           "JetHT+Run2018D-UL2018_MiniAODv2-v1+MINIAOD"
]

results = []
start = time.time()
if options.combined:
    cmd = 'python3 make_plots_data.py --tag={} --dataset={} -b'.format(options.tag, 'combined')
    results.append(pool.apply_async(call_makeplots, (cmd,)))
else:
    for sample in data_samples:
        cmd = 'python3 make_plots_data.py --tag={} --dataset={} -b'.format(options.tag, sample)
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
