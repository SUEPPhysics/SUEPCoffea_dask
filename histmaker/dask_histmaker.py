"""
Automatic histogram maker from the ntuples.
This script can make histograms for either a particular file, or a whole directory/sample of files (files here are intended as the ntuple hdf5 files).
It will fill all histograms that are part of the output dictionary (initialized in hist_defs.py), if it finds a matching variable in the ntuple dataframe.
(If doing an ABCD method, it will also fill each variable for each region.)
It can apply selections, blind, apply corrections and systematics, do ABCD method, make cutflows, and more.
The output histograms, cutflows, and metadata will be saved in a .root file.

e.g.
python make_hists.py --sample <sample> --output <output_tag> --tag <tag> --era <year> --isMC <bool> --doSyst <bool> --channel <channel>
"""

import argparse
import getpass
import logging
import os
import pickle
import subprocess
import sys
import uproot
import dask
import copy
import socket
import numpy as np
import pandas as pd
from dask import delayed
from typing import Optional, List, Tuple
from hist import Hist
from types import SimpleNamespace
from dask.distributed import Client, progress, LocalCluster, as_completed
from dask_jobqueue import SLURMCluster

sys.path.append("..")
import fill_utils
import hist_defs
import var_defs
from CMS_corrections import (
    GNN_syst,
    higgs_reweight,
    pileup_weight,
    track_killing,
    triggerSF,
)
import plotting.plot_utils as plot_utils

# TODO temp
dask.config.set({'logging.level': 'WARNING'})

class DaskHistMaker():

    def __init__(self, **kwargs) -> None:

        # TODO think about how we wanna structure this...
        self.options = kwargs.get("options", {})
        if self.options is {}:
            self.options = self.parseOptions()
        else:
            self.options = SimpleNamespace(**self.options)

        logging.basicConfig(level=logging.INFO)
        if self.options.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # TODO temp
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)

        # TODO make this required more explicitely?
        self.config = options["config"]

    def setupLocalClient(self, nThreads: int):

        client = LocalCluster(n_workers=2).get_client()
        logging.info(client)

        return client

    def setupSlurmClient(self):

        slurm_env = [
            'export DASK_DISTRIBUTED__COMM__ALLOWED_TRANSPORTS=["tcp://[::]:0"]',
            'export XRD_RUNFORKHANDLER=1',
            'export XRD_STREAMTIMEOUT=10',
            'export DASK_CONFIG=/home/submit/lavezzo/dask.yaml',
            'echo "Landed on $HOSTNAME"',
            f'source {os.getenv("HOME")}/.bashrc',
            f'conda activate SUEP',
            'echo "which python"',
            'which python',
            'cd /home/submit/lavezzo/SUEP/SUEPCoffea_dask/histmaker',
            'export PYTHONPATH=$PYTHONPATH:/home/submit/lavezzo/SUEP/SUEPCoffea_dask'
        ]

        extra_args=[
            "--output=/work/submit/lavezzo/SUEP/logs/dask_histmaker/dask_job_output_%j.out",
            "--error=/work/submit/lavezzo/SUEP/logs/dask_histmaker/dask_job_output_%j.err",
            "--partition=submit,submit-gpu",
        ]


        cluster = SLURMCluster(
            job_name="SUEP-dask-histmaker",
            cores=1,
            memory='8GB',
            scheduler_options={
                'dashboard_address': 9877,
                'host': socket.gethostname()
            },
            silence_logs="error",
            job_extra_directives=extra_args,
            job_script_prologue=slurm_env
        )

        cluster.scale(100)
        cluster.adapt(minimum=20, maximum=150)
        client = Client(cluster, heartbeat_interval='10s', timeout='10000s')

        logging.info(client)

        return client

    def parseOptions(self) -> Optional[dict]:

        parser = self.makeParser()
        options, _ = parser.parse_args()

        return vars(options)

    def makeParser(self, parser = None) -> argparse.ArgumentParser:

        if parser is None:
            parser = argparse.ArgumentParser(description="Famous Submitter")

        # tag for output histograms
        parser.add_argument("-o", "--output", type=str, help="output tag", required=True)
        # Where the files are coming from. Either provide:
        # 1. one filepath with -f
        # 2. ntuple --tag and --sample for something in dataDirLocal.format(tag, sample) (or dataDirXRootD with --xrootd 1)
        # 3. a directory of files: dataDirLocal (or dataDirXRootD with --xrootd 1)
        parser.add_argument(
            "-t",
            "--tag",
            type=str,
            default="IronMan",
            help="ntuples production tag",
            required=False,
        )
        parser.add_argument("--file", type=str, default="", help="Use specific input file")
        parser.add_argument(
            "--xrootd",
            type=int,
            default=0,
            help="Local data or xrdcp from hadoop (default=False)",
        )
        parser.add_argument(
            "--dataDirLocal",
            type=str,
            default="/data/submit//cms/store/user/" + getpass.getuser() + "/SUEP/{}/{}/",
            help="Local data directory",
        )
        parser.add_argument(
            "--dataDirXRootD",
            type=str,
            default=f"/cms/store/user/" + getpass.getuser() + "/SUEP/{}/{}/",
            help="XRootD data directory",
        )
        ## optional: call it with --merged = 1 to append a /merged/ to the paths in options 2 and 3
        parser.add_argument("--merged", type=int, default=0, help="Use merged files")
        # some required info about the files
        parser.add_argument("-e", "--era", type=str, help="era", required=True)
        parser.add_argument("--isMC", type=int, help="Is this MC or data", required=True)
        parser.add_argument(
            "--channel",
            type=str,
            help="Analysis channel: ggF, WH",
            required=True,
            choices=["ggF", "WH"],
        )
        # some analysis parameters you can toggle freely
        parser.add_argument(
            "--scouting", type=int, default=0, help="Is this scouting or no"
        )
        parser.add_argument("--doInf", type=int, default=0, help="make GNN plots")
        parser.add_argument(
            "--doABCD", type=int, default=0, help="make plots for each ABCD+ region"
        )
        parser.add_argument(
            "--doSyst",
            type=int,
            default=0,
            help="Run systematic up and down variations in addition to the nominal.",
        )
        parser.add_argument(
            "--blind", type=int, default=1, help="Blind the data (default=True)"
        )
        parser.add_argument(
            "-p",
            "--printEvents",
            action="store_true",
            help="Print out events that pass the selections, used in particular for eventDisplay.py.",
            required=False,
        )
        # other arguments
        parser.add_argument(
            "--saveDir",
            type=str,
            default=f"/data/submit/{getpass.getuser()}/SUEP/outputs/",
            help="Use specific output directory. Overrides default MIT-specific path.",
            required=False,
        )
        parser.add_argument(
            "--redirector",
            type=str,
            default="root://submit50.mit.edu/",
            help="xrootd redirector (default: root://submit50.mit.edu/)",
            required=False,
        )
        parser.add_argument(
            "--maxFiles",
            type=int,
            default=-1,
            help="Maximum number of files to process (default=None, all files)",
            required=False,
        )
        parser.add_argument(
            "--pkl",
            type=int,
            default=0,
            help="Write output to a pickle file instead of a root file (default=False)",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Run with verbose logging.",
            required=False,
        )
        return parser

    def get_filelist(self, sample: str = '') -> List[str]:

        logging.debug("Getting list of files.")
        if self.options.file:
            files = [self.options.file]
        elif self.options.xrootd:
            dataDir = (
                self.options.dataDirXRootD.format(self.options.tag, sample)
                if self.options.dataDirXRootD.count("{}") == 2
                else self.options.dataDirXRootD
            )
            if self.options.merged:
                dataDir += "/merged/"
            result = subprocess.check_output(["xrdfs", self.options.redirector, "ls", dataDir])
            result = result.decode("utf-8")
            files = result.split("\n")
            files = [f for f in files if len(f) > 0]
        else:
            dataDir = (
                self.options.dataDirLocal.format(self.options.tag, sample)
                if self.options.dataDirLocal.count("{}") == 2
                else self.options.dataDir
            )
            if self.options.merged:
                dataDir += "merged/"
            files = [dataDir + f for f in os.listdir(dataDir)]
        if self.options.maxFiles > 0:
            files = files[: self.options.maxFiles]
        files = [f for f in files if ".hdf5" in f]

        logging.debug("Found {} files.".format(len(files)))
        return files
    
    def write_output(self, sample: str, histograms: dict, metadata: dict = {}) -> None:
        """
        Writes the histograms and metadata to a .root file or a .pkl file, as specified by the options.
        """

        outFile = self.options.saveDir + "/" + sample + "_" + self.options.output
        if not os.path.exists(self.options.saveDir):
            os.makedirs(self.options.saveDir)
        if self.options.pkl:
            outFile += ".pkl"
            logging.info("Saving outputs to " + outFile)
            with open(outFile, "wb") as f:
                pickle.dump({"metadata": metadata, "hists": histograms}, f)
        else:
            outFile += ".root"
            logging.info("Saving outputs to " + outFile)
            with uproot.recreate(outFile) as froot:
                # write out metadata
                for k, m in metadata.items():
                    froot[f"metadata/{k}"] = str(m)

                # write out histograms
                for h, hist in histograms.items():
                    if len(hist.axes) > 3:
                        logging.warning(
                            f"Skipping {h} because it has more than 3 axes. This is not supported by .root files."
                        )
                        continue
                    froot[h] = hist

    def process_sample(self, client, sample) -> None:
        """
        For a given sample, process all the files and produce histograms.
        Cutflows and other metadata are stored in the metadata dictionary.
        Normalization and systematics needing the full histograms are applied.
        """

        logging.info("Processing sample: " + sample)

        commit, diff = fill_utils.get_git_info()
        metadata = {
            "git_commit": commit,
            "git_diff": diff,
            "ntuple_tag": self.options.tag,
            "analysis": self.options.channel,
            "isMC": int(self.options.isMC),
            "era": self.options.era,
            "sample": sample,
            "signal": (self.options.isMC) and (fill_utils.isSampleSignal(sample, self.options.era)),
            "xsec": 1,
            "gensumweight": 0,
            "lumi": 1,
            "nfiles": 0,
            "nfailed": 0,
        }

        files = self.get_filelist(sample)
        metadata['nfiles'] = len(files)
        if metadata['nfiles'] == 0:
            logging.error("No files found, exiting.")
            sys.exit(1)

        logging.info("Setup ready, filling histograms now.")
        # delayed_results = [delayed(self.process_file)(ifile, self.config, self.options, sample) for ifile in files]
        # results = client.compute(delayed_results)
        # progress(results)
        # results = client.gather(results)

        # TODO why was this complaining about being 1MiB?
        print(self.config)
        print(self.options)
        print(sample)

        return

        # Submit tasks and get Futures
        futures = [client.submit(self.process_file, ifile, self.config, self.options, sample) for ifile in files]

        total_gensumweight = 0
        nfailed = 0
        histograms = {}
        cutflow = {}

        # Iterate over futures as they complete
        for future in as_completed(futures):
            r = future.result()
            #results.append(result)
                

        # TODO: merge the outputs
        #logging.info("Merging outputs of files.")
        
        #for r in results:
            try:
                for h, hist in r[0].items():
                    if h in histograms:
                        histograms[h] += hist
                    else:
                        histograms[h] = hist
                for k, v in r[1].items():
                    if k in cutflow:
                        cutflow[k] += v
                    else:
                        cutflow[k] = v
                total_gensumweight += r[2]
            except Exception as e:
                logging.error(f"Failed to merge output: {e}")
                nfailed += 1
                continue

        if self.options.isMC and self.options.doSyst:
            logging.info("Applying symmetric systematics.")

            # do the tracks UP systematic
            histograms = track_killing.generate_up_histograms(histograms)

            for output_method in self.config.keys():
                if 'fGNNsyst' in self.config[output_method]:
                    # do the GNN systematic
                    GNN_syst.apply_GNN_syst(
                        histograms,
                        self.config[output_method]["fGNNsyst"],
                        self.config[output_method]["models"],
                        self.config[output_method]["GNNsyst_bins"],
                        self.options.era,
                        out_label=output_method,
                    )

        if self.options.isMC:
            logging.info("Applying normalization.")

            logging.debug(f"Found total_gensumweight {total_gensumweight}.")
            xsection = fill_utils.getXSection(sample, self.options.era, failOnKeyError=True)
            logging.debug(f"Found cross section x kr x br: {xsection}.")
            lumi = plot_utils.getLumi(self.options.era, scouting='scout' in self.options.channel)
            logging.debug(f"Found lumi: {lumi}.")

            if metadata['signal']:
                normalization = 1 / total_gensumweight
            else:
                # TODO: for testing
                total_gensumweight = 1
                normalization = xsection * lumi / total_gensumweight

            logging.info(f"Applying normalization: {normalization}.")
            histograms = fill_utils.apply_normalization(histograms, normalization)
            cutflow = fill_utils.apply_normalization(cutflow, normalization)

            # update metadata
            metadata['xsec'] = xsection
            metadata['lumi'] = lumi

        # update metadata
        metadata['gensumweight'] = total_gensumweight
        metadata['nfailed'] = nfailed
        if cutflow is not {}:
            for k, v in cutflow.items():
                metadata[k] = v

        self.write_output(sample, histograms, metadata)

    def process_file(self, ifile: str, config, options, sample: str = '') -> Tuple[dict, dict, dict]:
        """
        Read in ntuple hdf5 files and process each systematic variation to produce histograms and cutflows.
        Returns: histogram dictionary, cutflow dictionary, metadata dictionary.
        """

        file_cutflow, gensumweight = {}, 0
        file_histograms = copy.deepcopy(options.hists)
        
        # get the file
        df, ntuple_metadata, ntuple_hists = fill_utils.open_ntuple(
            ifile, redirector=options.redirector, xrootd=options.xrootd
        )
        if options.printEvents:
            # in the case we are interested in the run, lumi, event numbers, we also need to know the file they're from
            print(f"Opened file {ifile}")

        # check if file is corrupted
        if type(df) == int:
            logging.debug(f"File {ifile} is corrupted, skipping.")
            return {}, {}, 0

        # check sample consistency
        if sample != "" and ntuple_metadata.get("sample", False):
            if sample != ntuple_metadata["sample"]:
                raise Exception(
                        "This script should only run on one sample at a time. Found {} in ntuple metadata, and passed sample {}".format(
                            ntuple_metadata["sample"], sample
                        )
                    )

        # update the gensumweight
        if options.isMC:
            logging.debug(f"Found gensumweight {ntuple_metadata['gensumweight']}.")
            gensumweight = ntuple_metadata["gensumweight"]

        # update the cutflows
        if ntuple_metadata != 0 and any(["cutflow" in k for k in ntuple_metadata.keys()]):
            logging.debug("Found cutflows.")
            for k, v in ntuple_metadata.items():
                if "cutflow" in k:
                    if k not in file_cutflow.keys():
                        file_cutflow[k] = v
                    else:
                        file_cutflow[k] += v

        # update the ntuple histograms
        for hist_name, hist in ntuple_hists.items():
            logging.debug(f"Found histograms {hist_name}.")
            if hist_name in file_histograms:
                file_histograms[hist_name] += hist
            else:
                file_histograms[hist_name] = hist

        # check if any events are in the ntuple dataframe
        if "empty" in list(df.keys()):
            logging.debug("No events passed the selections, skipping.")
            return file_histograms, file_cutflow, gensumweight
        if df.shape[0] == 0:
            logging.debug("No events in file, skipping.")
            return file_histograms, file_cutflow, gensumweight
        
        # we might modify this for systematics, so make a copy
        config = copy.deepcopy(config)
        
        # TODO think about how to implement this. Either add to config by hand, or use the functions we already heve to do it
        # definitely does NOT need to be done every file!
        # config = fill_utils.get_track_killing_config(config)
        # config = fill_utils.get_jet_correction_config(config, syst)

        for config_tag, config_out in config.items():

            logging.debug(f"\tUsing configuration {config_tag}.")

            logging.debug(f"\tRunning nominal variation")
            self.plot_variation(df, '', config_out, config_tag, file_histograms, file_cutflow, ntuple_metadata)

            for syst in config_out.get("syst", []):
                logging.debug(f"Running syst {syst}")
                self.plot_variation(df, syst, config_out, config_tag, file_histograms, file_cutflow, ntuple_metadata)

        # remove file at the end of loop
        if options.xrootd:
            fill_utils.close_ntuple(ifile)

        return file_histograms, file_cutflow, gensumweight

    
    def plot_variation(
            self, 
            df:pd.DataFrame, variation:str,  config_out:dict, config_tag:str,
            histograms:dict, cutflow:dict = {}, metadata:dict = {}
        ):

        df_plot = df.copy()

        # rename output method if we have applied a systematic
        if len(variation) > 0:
            config_tag = config_tag + "_" + variation

        logging.debug(f"\tFilling variation {variation} ({config_tag}).")

        # apply event weights
        df_plot = self.apply_event_weights(df_plot, variation, metadata["sample"], config_tag)

        # prepare the DataFrame for plotting: blind, selections, new variables
        df_plot = fill_utils.prepare_DataFrame(
            df_plot,
            config_out,
            config_tag,
            isMC=self.options.isMC,
            blind=self.options.blind,
            cutflow=cutflow,
            output=histograms,
        )

        # if there are no events left after selections, no need to fill histograms
        if (df_plot is None) or (df_plot.shape[0] == 0):
            return histograms, cutflow, metadata

        # print out events that pass the selections, if requested
        if self.options.printEvents:
            print("Events passing selections for", config_tag)
            for index, row in df_plot.iterrows():
                print(
                    f"{int(row['event'])}, {int(row['run'])}, {int(row['luminosityBlock'])}"
                )

        # TODO fix this
        # if no output histograms are defined for this method, skip it
        # if len([h for h in config_out["hists"] if h in histograms.keys()]) == 0:
        #     logging.warning(f"\tNo histograms defined for configuration {config_tag}, skipping filling histograms.")
        #     return histograms, cutflow, metadata

        # auto fill all histograms
        fill_utils.auto_fill(
            df_plot,
            histograms,
            config_out,
            config_tag,
            isMC=self.options.isMC,
            do_abcd=self.options.doABCD,
        )

        return histograms, cutflow, metadata

    def apply_event_weights(self, df: pd.DataFrame, variation: str, sample: str = '', config_tag: str = '') -> pd.DataFrame:

        # apply event weights
        if self.options.isMC:

            df["event_weight"] = df["genweight"].to_numpy()

            # 1) pileup weights
            puweights, puweights_up, puweights_down = pileup_weight.pileup_weight(
                self.options.era
            )
            pu = pileup_weight.get_pileup_weights(
                df, variation, puweights, puweights_up, puweights_down
            )
            df["event_weight"] *= pu

            # 2) PS weights
            if "PSWeight" in variation and variation in df.keys():
                df["event_weight"] *= df[variation]

            # 3) prefire weights
            if (self.options.era in ["2016apv", "2016", "2017"]):
                if "prefire" in variation and variation in df.keys():
                    df["event_weight"] *= df[variation]
                else:
                    df["event_weight"] *= df["prefire_nom"]

            # 4) TriggerSF weights
            if self.options.channel == "ggF":
                if self.options.scouting != 1:
                    (
                        trig_bins,
                        trig_weights,
                        trig_weights_up,
                        trig_weights_down,
                    ) = triggerSF.triggerSF(self.options.era)
                    trigSF = triggerSF.get_trigSF_weight(
                        df,
                        variation,
                        trig_bins,
                        trig_weights,
                        trig_weights_up,
                        trig_weights_down,
                    )
                    df["event_weight"] *= trigSF
                else:
                    trigSF = triggerSF.get_scout_trigSF_weight(
                        np.array(df["ht"]).astype(int), variation, self.options.era
                    )
                    df["event_weight"] *= trigSF

            # 5) Higgs_pt weights
            if "mS125" in sample:
                (
                    higgs_bins,
                    higgs_weights,
                    higgs_weights_up,
                    higgs_weights_down,
                ) = higgs_reweight.higgs_reweight(df["SUEP_genPt"])
                higgs_weight = higgs_reweight.get_higgs_weight(
                    df,
                    variation,
                    higgs_bins,
                    higgs_weights,
                    higgs_weights_up,
                    higgs_weights_down,
                )
                df["event_weight"] *= higgs_weight

             # 8) b-tag weights. These have different values for each event selection
            if self.options.channel == 'WH' and self.options.isMC:
                if 'btag' in variation.lower():
                    btag_weights = variation
                else:
                    btag_weights = 'bTagWeight_nominal'
                btag_weights += "_" + config_tag
                if btag_weights not in df.keys():
                    logging.warning(f"btag weights {btag_weights} not found in DataFrame. Not applying them.")
                else:
                    df['event_weight'] *= df[btag_weights]

        # data
        else:
            df["event_weight"] = np.ones(df.shape[0])

        return df

            

    
if __name__ == "__main__":

    config = {
        "CRWJ": {
            "input_method": "HighestPT",
            "method_var": "SUEP_nconst_HighestPT",
            "SR": [
                ["SUEP_S1_HighestPT", ">=", 0.3],
                ["SUEP_nconst_HighestPT", ">=", 40],
            ],
            "selections": [
                "WH_MET_pt > 30",
                "W_pt > 40",
                "W_mt < 130",
                "W_mt > 30",
                "bjetSel == 1",
                "deltaPhi_SUEP_W > 1.5",
                "deltaPhi_SUEP_MET > 1.5",
                "deltaPhi_lepton_SUEP > 1.5",
                "ak4jets_inSUEPcluster_n_HighestPT >= 1",
                "W_SUEP_BV < 2",
                "deltaPhi_minDeltaPhiMETJet_MET > 1.5",
                "SUEP_S1_HighestPT < 0.3",
                "SUEP_nconst_HighestPT < 40",
                "lepton_pt > 100"
            ],
            "syst":  [
                "puweights_up",
                "puweights_down"
            ], # a list of systematics
        },
    }
    
    options = {
        'isMC': 0,
        'channel': 'WH',
        'era': '2018',
        'tag': 'WH_7_20',
        'output': 'testDaskHistMaker',
        'doSyst': 0,
        'verbose': 1,
        'xrootd': 0,
        'saveDir': '/work/submit/lavezzo/SUEP/outputs/',
        'config': config,
        'file': None,
        'dataDirLocal': '/data/submit/cms/store/user/lavezzo/SUEP/{}/{}/',
        'dataDirXRootD': '/cms/store/user/lavezzo/SUEP/{}/{}/',
        'merged': 0,
        'maxFiles': -1,
        'pkl': 1,
        'printEvents': False,
        'scouting': 0,
        'doInf': 0,
        'doABCD': 0,
        'blind': 1,
        'redirector': 'root://submit50.mit.edu/',
        'hists': {}
    }

    # TODO this is of course stupid
    var_defs.initialize_new_variables('CRWJ', SimpleNamespace(**options), config['CRWJ'])
    hist_defs.initialize_histograms(options['hists'], 'CRWJ', SimpleNamespace(**options), config['CRWJ'])
    
    histmaker = DaskHistMaker(options=options)
    #client = histmaker.setupLocalClient(2)
    client = histmaker.setupSlurmClient()
    histmaker.process_sample(client, 'SingleMuon+Run2018A-UL2018_MiniAODv2-v3+MINIAOD')
    client.close()