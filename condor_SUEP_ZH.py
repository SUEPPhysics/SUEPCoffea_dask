import argparse
import os

# Import coffea specific features
from coffea import processor
from coffea.processor import Runner, futures_executor, run_uproot_job

# SUEP Repo Specific
from workflows import SUEP_coffea_ZH, merger

# Begin argparse
parser = argparse.ArgumentParser("")
parser.add_argument("--isMC", type=int, default=1, help="")
parser.add_argument("--jobNum", type=int, default=1, help="")
parser.add_argument("--era", type=str, default="2018", help="")
parser.add_argument("--doSyst", type=int, default=1, help="")
parser.add_argument("--infile", required=True, type=str, default=None, help="")
parser.add_argument("--dataset", type=str, default="X", help="")
parser.add_argument("--nevt", type=str, default=-1, help="")
parser.add_argument(
    "--SR",
    action="store_true",
    default=False,
    help="Save only SR-level results. For quick checks",
)
parser.add_argument(
    "--doOF",
    action="store_true",
    default=False,
    help="Do emu final state instead of ee+mumu",
)
parser.add_argument(
    "--isDY",
    action="store_true",
    default=False,
    help="Activate to save the gen-level of the Z pT, needed to clean the overlap in DY samples",
)


options = parser.parse_args()

out_dir = os.getcwd()
modules_era = []

modules_era.append(
    SUEP_coffea_ZH.SUEP_cluster_ZH(
        isMC=options.isMC,
        era=str(options.era),
        scouting=0,
        do_syst=options.doSyst,
        syst_var="",
        sample=options.dataset,
        weight_syst="",
        flag=False,
        output_location=out_dir,
        SRonly=options.SRonly,
        doOF=options.doOF,
        isDY=options.isDY,
    )
)

for instance in modules_era:
    runner = processor.Runner(
        executor=processor.FuturesExecutor(compression=None, workers=1),
        schema=processor.NanoAODSchema,
        xrootdtimeout=60,
        chunksize=100000000,
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
