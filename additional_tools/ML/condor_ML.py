import argparse
import os

# Import coffea specific features
from coffea import processor
from coffea.processor import futures_executor, run_uproot_job

# SUEP Repo Specific
from workflows import ML_coffea, merger

# Begin argparse
parser = argparse.ArgumentParser("")
parser.add_argument("--isMC", type=int, default=1, help="")
parser.add_argument("--jobNum", type=int, default=1, help="")
parser.add_argument("--era", type=str, default="2018", help="")
parser.add_argument("--doSyst", type=int, default=1, help="")
parser.add_argument("--doInf", type=int, default=1, help="")
parser.add_argument("--infile", type=str, default=None, help="")
parser.add_argument("--dataset", type=str, default="X", help="")
parser.add_argument("--nevt", type=str, default=-1, help="")
options = parser.parse_args()

out_dir = os.getcwd()
modules_era = []

modules_era.append(
    ML_coffea.ML_cluster(
        isMC=options.isMC,
        era=int(options.era),
        scouting=0,
        do_syst=options.doSyst,
        syst_var="",
        sample=options.dataset,
        weight_syst="",
        flag=False,
        do_inf=False,
        output_location=out_dir,
    )
)

for instance in modules_era:
    runner = processor.Runner(
        executor=processor.FuturesExecutor(compression=None, workers=1),
        schema=processor.NanoAODSchema,
        xrootdtimeout=60,
        chunksize=10000,
    )

    runner.automatic_retries(
        retries=3,
        skipbadfiles=False,
        func=runner.run,
        fileset={options.dataset: [options.infile]},
        treename="Events",
        processor_instance=instance,
    )

    merger.merge(options, pattern="condor_*.hdf5", outFile="out.hdf5")


merger.merge_ML(options)
