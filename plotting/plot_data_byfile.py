import os
import subprocess
import shlex
import argparse
import multiprocessing
from multiprocessing.pool import ThreadPool
import time
import getpass

parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan"  , help="Production tag", required=True)
options = parser.parse_args()

def call_makeplots(cmd):
    """ This runs in a separate thread. """
    print("----[%] :", cmd)
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)

pool = ThreadPool(multiprocessing.cpu_count())

username = getpass.getuser()
dataDir = "/mnt/T3_US_MIT/hadoop/scratch/{}/SUEP/{}/".format(username,options.tag)
files = [f for f in os.listdir(dataDir)]
results = []
start = time.time()
for i, file in enumerate(files):
    cmd = 'python3 make_plot_data.py --tag={} --file={} --number={} -b'.format(options.tag, file, i)
    results.append(pool.apply_async(call_makeplots, (cmd,)))
    if i > 100: break
    
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
