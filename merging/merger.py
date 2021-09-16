import os, glob
from array import array
import sys
from termcolor import colored



indir = "/mnt/hadoop/scratch/freerc/SUEP/Wonderland/"
outdir = "/mnt/hadoop/scratch/freerc/SUEP/Wonderland/merged"

def main():
    pattern = "QCD"
    for sample in os.listdir(indir):
        print colored(" -- " + sample, "blue")
        if "merged" in sample:
            continue
        #if pattern not in sample:
        #    continue
        in_files = glob.glob("{indir}{sample}/*.root".format(sample=sample, indir=indir))
        str1 = " "
        in_files = str1.join(in_files)
        out_file = "{outdir}/{sample}.root".format(sample=sample, outdir=outdir)
        if len(in_files) == 0:
            print colored(" -- [warning] empty directory for " + sample, "red")
            continue
        os.system("python haddnano.py {} {}".format("out.root",in_files))
        os.system("mv out.root {}".format(out_file))

if __name__ == "__main__":
   main()
