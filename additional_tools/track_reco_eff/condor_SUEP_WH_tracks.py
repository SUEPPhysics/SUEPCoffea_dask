import argparse
import os
import sys
import h5py
import hist
sys.path.append("../../")

# Import coffea specific features
from coffea import processor
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster

# SUEP Repo Specific
import SUEP_coffea_WH_tracks
from workflows.utils import output_utils, pandas_utils
import pickle


def form_ntuple(options, output):
    """
    Extract the dataframes from the processor output
    We expect this to have the shape: {"variation1": {"vars": df, ...}, "variation2": {"vars": df, ...}, ...}
    Output is: [df1, df2, ...], ["vars_variation1", "vars_variation2", ...]
    These will be saved in the hdf5 as
        "vars_variation1"
        "vars_variation2"
        ...
    """
    dfs = []
    variations = output["out"][options.dataset].keys()
    for var in variations:
        dfs.append(pandas_utils.format_dataframe(output["out"][options.dataset][var]["vars"].value))
    return dfs, ["vars_" + var if var != "nominal" else "vars" for var in variations]

def form_metadata(options, output):
    """
    Extract the metadata from the processor output
    We expect this to have the shape: {variation1: {era: 2018, mc: 1, sample: X, ...}, ...}
    Output is: {era_variation1: 2018, era_variation2: 2018, ..., mc_variation1: 1, mc_variation2: 1, ..., sample_variation1: X, sample_variation2: X, ...}
    This will be saved in the hdf5 as
        metadata
    """
    metadata = dict(
        era=options.era,
        mc=options.isMC,
        sample=options.dataset,
    )
    variations = output["out"][options.dataset].keys()
    for var in variations:
        metadata.update(
            {
                "_".join(filter(None, [key, var])): output["out"][options.dataset][var][key]
                for key in output["out"][options.dataset][var].keys()
                if type(output["out"][options.dataset][var][key]) is processor.value_accumulator
            }
        )
    metadata = pandas_utils.format_metadata(metadata)
    return metadata


def form_hists(options, output):
    """
    Extract the histograms from the processor output
    We expect this to have the shape: {variation1: {hist_name1: hist1, ...}, variation2: {hist_name1: hist1, ...}, ...}
    Output is: [{hist_name1: hist1, ...}, {hist_name1: hist1, ...}, ...], [hists_variation1, hists_variation2, ...]
    These will be saved in the hdf5 as
        hists_variation1/hist_name1
        hists_variation1/hist_name2
        ...
        hists_variation2/hist_name1
        ...
    """
    hists = []
    variations = output["out"][options.dataset].keys()
    for var in variations:
        hists_var = {}
        for key in output["out"][options.dataset][var].keys():
            if type(output["out"][options.dataset][var][key]) is hist.Hist:
                hists_var[key] = output["out"][options.dataset][var][key]
        hists.append(hists_var)
    return hists, ["hists_" + var if var != "nominal" else "hists" for var in variations]


def main():
    # Begin argparse
    parser = argparse.ArgumentParser("")
    parser.add_argument("--isMC", type=int, default=1, help="")
    parser.add_argument("--jobNum", type=int, default=1, help="")
    parser.add_argument("--era", type=str, default="2018", help="")
    parser.add_argument("--doSyst", type=int, default=0, help="")
    parser.add_argument("--infile", required=True, type=str, default=None, help="")
    parser.add_argument(
        "--outfile",
        "-f",
        default="out.pkl",
        help="Output file name.",
        type=str,
    )
    parser.add_argument(
        "--output_location",
        "-o",
        default=os.getcwd(),
        help="Path to output directory, can be xrootd or local",
        type=str,
    )
    parser.add_argument("--dataset", type=str, default="X", help="")
    parser.add_argument("--maxChunks", type=int, default=None, help="")
    parser.add_argument("--chunkSize", type=int, default=100, help="")
    parser.add_argument("-n", type=int, default=100, help="")
    parser.add_argument(
        "--doInf",
        type=str,
        default=None,
        help="Only added for compatibility with kraken_run.py",
    )
    options = parser.parse_args()

    modules_era = []

    modules_era.append(
        SUEP_coffea_WH_tracks.SUEP_cluster_WH(
            isMC=options.isMC,
            era=str(options.era),
            do_syst=options.doSyst,
            sample=options.dataset,
            flag=False,
            output_location=options.output_location,
        )
    )

    # slurm_env = [
    #     'export DASK_DISTRIBUTED__COMM__ALLOWED_TRANSPORTS=["tcp://[::]:0"]',
    #     "export XRD_RUNFORKHANDLER=1",
    #     "export XRD_STREAMTIMEOUT=10",
    #     'echo "Landed on $HOSTNAME"',
    #     f'echo "source {os.getenv("HOME")}/.bashrc"',
    #     f'source {os.getenv("HOME")}/.bashrc',
    #     f'echo "cd {os.path.dirname(os.path.abspath(__file__))}"',
    #     f"cd {os.path.dirname(os.path.abspath(__file__))}",
    #     f'echo "export PYTHONPATH=$PYTHONPATH:{os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))}"',
    #     f"export PYTHONPATH=$PYTHONPATH:{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}",
    #     'echo "conda activate suep"',
    #     f"conda activate suep",
    #     'echo "which python"',
    #     "which python",
    #     'echo "Worker environment setup done."',
    # ]
    # logDir = "/work/submit/lavezzo/SUEP/logs/dask_track_eff/"
    # if not os.path.exists(logDir):
    #     os.makedirs(logDir)
    # extra_args = [
    #     f"--output={logDir}/job_output_%j.out",
    #     f"--error={logDir}/job_output_%j.err",
    #     "--partition=submit,submit-gpu",
    # ]

    # cluster = SLURMCluster(
    #     job_name="tracks",
    #     cores=1,
    #     walltime="5:00:00",
    #     memory="4GB",
    #     scheduler_options={
    #         "dashboard_address": "1776",
    #     },
    #     silence_logs="warning",
    #     job_extra_directives=extra_args,
    #     job_script_prologue=slurm_env,
    # )
    # cluster.scale(options.n)
    # client = Client(cluster)

    cluster = LocalCluster(
        n_workers=options.n,
        #threads_per_worker=1,
        dashboard_address="1776",
    )

    client = Client(cluster)
    for instance in modules_era:
        
        runner = processor.Runner(
            executor=processor.DaskExecutor(client=client),
            schema=processor.NanoAODSchema,
            xrootdtimeout=120,
            #chunksize=100,
            chunksize=options.chunkSize,
            maxchunks=options.maxChunks,
        )

        infiles = options.infile
        # if the infile is a dir, list it
        if os.path.isdir(infiles):
            infiles = [os.path.join(infiles, f) for f in os.listdir(infiles) if '.root' in f]
        else:
            infiles = [infiles]

        output = runner.automatic_retries(
            retries=0,
            skipbadfiles=False,
            func=runner.run,
            fileset={options.dataset: infiles},
            treename="Events",
            processor_instance=instance,
        )

        # format the desired data from the processor output
        hists, hist_names = form_hists(options, output)

        # Save histograms to a pkl file
        hist_output = {name: hist for name, hist in zip(hist_names, hists)}
        with open(options.outfile, 'wb') as f:
            pickle.dump(hist_output, f)

    client.close()

if __name__ == "__main__":
    main()
