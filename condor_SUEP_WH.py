import argparse
import os

import h5py
import hist

# Import coffea specific features
from coffea import processor

# SUEP Repo Specific
from workflows import SUEP_coffea_WH
from workflows.utils import output_utils, pandas_utils


def form_ntuple(options, output):
    df = pandas_utils.format_dataframe(output["out"][options.dataset]["vars"].value)
    return df


def form_metadata(options, output):
    metadata = dict(
        era=options.era,
        mc=options.isMC,
        sample=options.dataset,
    )
    metadata.update(
        {
            key: output["out"][options.dataset][key]
            for key in output["out"][options.dataset].keys()
            if type(output["out"][options.dataset][key]) is processor.value_accumulator
        }
    )
    metadata = pandas_utils.format_metadata(metadata)

    return metadata


def form_hists(options, output):
    hists = {}
    for key in output["out"][options.dataset].keys():
        if type(output["out"][options.dataset][key]) is hist.Hist:
            hists[key] = output["out"][options.dataset][key]
    return hists


def main():
    # Begin argparse
    parser = argparse.ArgumentParser("")
    parser.add_argument("--isMC", type=int, default=1, help="")
    parser.add_argument("--jobNum", type=int, default=1, help="")
    parser.add_argument("--era", type=str, default="2018", help="")
    parser.add_argument("--doSyst", type=int, default=1, help="")
    parser.add_argument("--infile", required=True, type=str, default=None, help="")
    parser.add_argument(
        "--outfile",
        "-f",
        default="out.hdf5",
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
    parser.add_argument("--chunkSize", type=int, default=100000, help="")
    parser.add_argument(
        "--doInf",
        type=str,
        default=None,
        help="Only added for compatibility with kraken_run.py",
    )
    options = parser.parse_args()

    modules_era = []

    modules_era.append(
        SUEP_coffea_WH.SUEP_cluster_WH(
            isMC=options.isMC,
            era=str(options.era),
            do_syst=options.doSyst,
            sample=options.dataset,
            flag=False,
            output_location=options.output_location,
            dropNonMethodEvents=True,
            storeJetsInfo=False,
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

        # format the desired data from the processor output
        df = form_ntuple(options, output)
        metadata = form_metadata(options, output)
        hists = form_hists(options, output)

        # save everything to hdf5
        pandas_utils.save_dfs(
            instance, [df], ["vars"], options.outfile, metadata=metadata
        )
        output_utils.add_hists(options.outfile, hists)

        # write out the hdf5 to the output_location
        output_utils.dump_table(
            fname=options.outfile, location=instance.output_location
        )


if __name__ == "__main__":
    main()
