import argparse
import os

# Import coffea specific features
from coffea.processor import futures_executor, run_uproot_job

# SUEP Repo Specific
from coffea import processor
from workflows import SUEP_coffea, merger, root_rewrite

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
        era=int(options.era),
        scouting=1,
        do_syst=options.doSyst,
        syst_var="",
        sample=options.dataset,
        weight_syst="",
        flag=False,
        do_inf=options.doInf,
        output_location=out_dir,
    )
)

processor.NanoAODSchema.mixins["PFcand"] = "PFCand"
for instance in modules_era:
    output = run_uproot_job(
        {instance.sample: ["rewrite.root"]},
        treename="tree",
        processor_instance=instance,
        executor=futures_executor,
        executor_args={
            "workers": 1,
            "schema": processor.NanoAODSchema,
            "xrootdtimeout": 10,
        },
        chunksize=100000,
    )

os.system("rm rewrite.root")
merger.merge(options)
