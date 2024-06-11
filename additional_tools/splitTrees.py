#!/usr/bin/env python

"""
Split SUEP samples by complete parameter points (mPhi, T, decay mode).

Author: Carlos Erice Cid
"""


import os

import ROOT


def splitfile(inputs):
    fname, options = inputs[0], inputs[1]
    allpoints = {}
    if not os.path.exists(options.output):
        os.system("mkdir -p " + options.output)
    f = ROOT.TFile.Open(fname, "read")
    t = f.Events
    print("Total events in %s: %d" % (fname, t.GetEntries()))

    # First we keep nothing
    t.SetBranchStatus("*", 0)  # Speed up by only using useful branches
    # split using GenModel information
    t.SetBranchStatus("GenModel*", 1)  # Speed up by only using useful branches
    list_branches = [key.GetName() for key in t.GetListOfBranches()]
    for l in list_branches:
        if "GenModel" in l:
            name = l.replace("GenModel_", "")
            allpoints[name] = ROOT.TEventList(name, name)

    # Now we fill up the TEventList
    for nev in range(t.GetEntries()):
        if nev % 10000 == 1:
            print("Gen-Scanning event %d/%d" % (nev, t.GetEntries()))
        t.GetEntry(nev)
        for key in allpoints.keys():
            if getattr(t, "GenModel_" + key):
                allpoints[key].Enter(nev)

    for m in sorted(allpoints.keys()):
        print("-------%s: %d events" % (m, allpoints[m].GetN()))

    # Now we reactivate all branches so we save the whole tree!
    t.SetBranchStatus("*", 1)
    for drop in options.drop:
        t.SetBranchStatus(drop, 0)  # Except those we don't want
    for keep in options.keep:
        t.SetBranchStatus(
            keep, 1
        )  # Just in case we want to do some regexp with the previous step

    # The actual saving
    for m, elist in allpoints.items():
        if not os.path.isdir(options.output + m):
            os.mkdir(options.output + m)

        output = (
            options.output
            + m
            + "/"
            + fname.split("/")[-1].replace(".root", "_" + m + ".root")
        )
        print()
        print("writing file for this signal point to:", output)
        print()
        if os.path.exists(output):
            raise RuntimeError("Output file already exists")
        fout = ROOT.TFile(output, "recreate")
        fout.cd()
        t.SetEventList(elist)
        out = t.CopyTree("1")
        fout.WriteTObject(out, "Events")
        fout.Close()
    return allpoints.keys()


def haddfiles(inputs):
    outdir = inputs[0]
    name = inputs[1]
    os.system(
        "python haddnano.py %s/%s_merged.root %s/*%s*root"
        % (outdir, name, outdir, name)
    )


if __name__ == "__main__":
    from optparse import OptionParser

    parser = OptionParser(usage="%prog [options] outputDir inputDirs")
    parser.add_option(
        "-D",
        "--drop",
        dest="drop",
        type="string",
        default=["GenModel_*"],
        action="append",
        help="Branches to drop, as per TTree::SetBranchStatus. Default is to just drop the 'GenModel' ones.",
    )
    parser.add_option(
        "-K",
        "--keep",
        dest="keep",
        type="string",
        default=[],
        action="append",
        help="Branches to keep, as per TTree::SetBranchStatus. Default is all, but can we used to reactive after kdrop.",
    )
    parser.add_option(
        "--inputFiles",
        dest="inputFiles",
        type="string",
        default=None,
        help="txt file with list of input files (one on each line)",
    )
    parser.add_option(
        "--output",
        dest="output",
        type="string",
        default=None,
        help="Output folder containing the .root files after the splitting",
    )
    parser.add_option(
        "--jobs",
        dest="jobs",
        type="int",
        default=1,
        help="How many cores to take (default = 1, sequential running).",
    )
    parser.add_option(
        "--hadd",
        dest="hadd",
        action="store_true",
        default=False,
        help="If activated, run hadd over split chunks to get merged .root files.",
    )

    (options, args) = parser.parse_args()

    allInputFiles = []
    with open(options.inputFiles) as filelist:
        allInputFilesRaw = filelist.readlines()

    allInputFiles = [line.strip() for line in allInputFilesRaw]

    print("Will write selected trees to " + options.output)
    allpoints = []
    if options.jobs == 1:
        for f in allInputFiles:
            allpoints += splitfile([f, options])
    else:
        allInputs = [[f, options] for f in allInputFiles]
        import time
        from contextlib import closing
        from multiprocessing import Pool

        with closing(Pool(options.jobs)) as p:
            retlist1 = p.map_async(splitfile, allInputs, 1)
            while not retlist1.ready():
                time.sleep(0.001)
            retlist1 = retlist1.get()
            p.close()
            p.join()
            p.terminate()
        for r in retlist1:
            allpoints += r
    allpoints = list(dict.fromkeys(allpoints))
    if options.hadd:
        print("Will now hadd the split chunks")
        if options.jobs == 1:
            for p in allpoints:
                haddfiles([options.output, p])
        else:
            allInputs = [[options.output, p] for p in allpoints]
            with closing(Pool(options.jobs)) as p:
                retlist1 = p.map_async(haddfiles, allInputs, 1)
                while not retlist1.ready():
                    time.sleep(0.001)
                retlist1 = retlist1.get()
                p.close()
                p.join()
                p.terminate()
