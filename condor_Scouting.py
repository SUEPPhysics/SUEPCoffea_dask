import argparse
import os

# Import coffea specific features
from coffea import processor
from coffea.processor import Runner, futures_executor, run_uproot_job

from workflows import SUEP_coffea

# SUEP Repo Specific
from workflows.utils import merger, root_rewrite

# Begin argparse
parser = argparse.ArgumentParser("")
parser.add_argument("--isMC", type=int, default=1, help="")
parser.add_argument("--jobNum", type=int, default=1, help="")
parser.add_argument("--era", type=str, default="2018", help="")
parser.add_argument("--doSyst", type=int, default=1, help="")
parser.add_argument("--doInf", type=int, default=0, help="")
parser.add_argument("--infile", type=str, default=None, help="")
parser.add_argument("--dataset", type=str, default="X", help="")
parser.add_argument("--nevt", type=str, default=-1, help="")
options = parser.parse_args()

out_dir = os.getcwd()
modules_era = []

root_rewrite.rewrite(options.infile)

modules_era.append(
    SUEP_coffea.SUEP_cluster(
        isMC=options.isMC,
        era=options.era,
        scouting=1,
        do_syst=options.doSyst,
        syst_var="",
        sample=options.dataset,
        weight_syst="",
        flag=False,
        do_inf=options.doInf,
        output_location=out_dir,
        accum="pandas_merger",
    )
)

processor.NanoAODSchema.mixins["PFcand"] = "PFCand"
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
        fileset={options.dataset: ["rewrite.root"]},
        treename="tree",
        processor_instance=instance,
    )

    merger.merge(options, pattern="ntuple_*.hdf5", outFile="out.hdf5")

os.system("rm rewrite.root")
