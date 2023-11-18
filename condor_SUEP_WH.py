import argparse
import os

# Import coffea specific features
import coffea
from coffea import processor
from coffea.processor import Runner, futures_executor, run_uproot_job

# SUEP Repo Specific
from workflows import SUEP_coffea_WH, pandas_utils

# Begin argparse
parser = argparse.ArgumentParser("")
parser.add_argument("--isMC", type=int, default=1, help="")
parser.add_argument("--jobNum", type=int, default=1, help="")
parser.add_argument("--era", type=str, default="2018", help="")
parser.add_argument("--doSyst", type=int, default=1, help="")
parser.add_argument("--infile", required=True, type=str, default=None, help="")
parser.add_argument("--dataset", type=str, default="X", help="")
parser.add_argument("--maxChunks", type=int, default=None, help="")
parser.add_argument("--chunkSize", type=int, default=100000, help="")
parser.add_argument("--doInf", type=str, default=-1, help="")
options = parser.parse_args()

out_dir = os.getcwd()
modules_era = []

modules_era.append(
    SUEP_coffea_WH.SUEP_cluster_WH(
        isMC=options.isMC,
        era=str(options.era),
        scouting=0,
        do_syst=options.doSyst,
        syst_var="",
        sample=options.dataset,
        weight_syst="",
        flag=False,
        output_location=out_dir,
        accum="pandas_merger",
    )
)

for instance in modules_era:
    runner = processor.Runner(
        executor=processor.FuturesExecutor(compression=None, workers=1),
        schema=processor.NanoAODSchema,
        xrootdtimeout=120,
        chunksize=options.chunkSize,
        maxchunks=options.maxChunks,
    )

    output = runner.automatic_retries(
        retries=3,
        skipbadfiles=False,
        func=runner.run,
        fileset={options.dataset: [options.infile]},
        treename="Events",
        processor_instance=instance,
    )

    # save output
    df = output["out"][options.dataset]["vars"].value
    metadata = dict(
        era=options.era,
        mc=options.isMC,
        sample=options.dataset,
    )
    metadata.update(
        {
            key: output["out"][options.dataset][key]
            for key in output["out"][options.dataset].keys()
            if key != "vars"
        }
    )
    pandas_utils.save_dfs(instance, [df], ["vars"], "out.hdf5", metadata=metadata)
