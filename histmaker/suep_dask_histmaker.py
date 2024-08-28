"""
Author: Luca Lavezzo
Date: August 2024
"""

import logging
import os
import pickle
import subprocess
import sys
import uproot
import copy
import json
import numpy as np
import pandas as pd
from dask import delayed
from typing import List
from types import SimpleNamespace
from dask.distributed import Client, Future
from coffea.processor import value_accumulator

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
from dask_histmaker import BaseDaskHistMaker


class SUEPDaskHistMaker(BaseDaskHistMaker):
    """
    Dask HistMaker for SUEP analyses.
    Processes hdf5 ntuples containing event information and produces histograms.
    See histmaker/README.md for more information.
    """

    def __init__(self, options: dict, **kwargs) -> None:

        super().__init__(**kwargs)

        self.options = self.get_options(options)
        self.config = kwargs.get("config", {})
        self.hists = kwargs.get("hists", {})

        logging.basicConfig(level=logging.ERROR) # TODO quiet dask_distributed and dask_joqueue, this doesn't seem to work
        self.logger = logging.getLogger(self.__class__.__name__) # logging for this class only
        self.logger.setLevel(level=logging.INFO)
        if self.options.verbose:
            self.logger.setLevel(logging.DEBUG)
            #TODO debug
            #pass

    def get_options(self, options: dict) -> SimpleNamespace:
        """
        Initialize options for processing.
        """

        self.default_options = {
            'output': 'daskHistMaker',
            'doSyst': 0,
            'verbose': 0,
            'xrootd': 0,
            'saveDir': '/ceph/submit/data/user/l/'+os.environ['USER']+'/SUEP/outputs/',
            'logDir': '/work/submit/'+os.environ['USER']+'/SUEP/logs/',
            'file': None,
            'dataDirLocal': '/data/submit/cms/store/user/'+os.environ['USER']+'/SUEP/{}/{}/',
            'dataDirXRootD': '/cms/store/user/'+os.environ['USER']+'/SUEP/{}/{}/',
            'merged': 0,
            'maxFiles': -1,
            'pkl': 1,
            'printEvents': False,
            'scouting': 0,
            'doInf': 0,
            'doABCD': 0,
            'blind': 1,
            'redirector': 'root://submit50.mit.edu/',
        }
        self.mandatory_options = ['era', 'tag', 'channel', 'isMC']
        self.options = self.default_options
        self.options.update(options)
        for opt in self.mandatory_options:
            if opt not in self.options:
                raise ValueError(f"Missing mandatory option {opt}.")
            
        return SimpleNamespace(**self.options)

    def preprocess_sample(self, sample: str) -> None:
        """
        Not needed for now.
        """
        pass

    def process_sample(self, client: Client, sample: str) -> List[Future]:
        """
        For a given sample, fetch list of and process all the files.
        """

        self.logger.info("Processing sample: " + sample)

        files = self.get_filelist(sample)
        if len(files) == 0:
            self.logger.error("No files found, exiting.")
            return []
        
        self.logger.debug("Creating futures for sample " + sample)

        futures = [client.submit(self.process_file, ifile, sample, self.hists, self.config, self.options) for ifile in files]

        del files

        return futures
    
    def postprocess_sample(self, sample: str, output: dict) -> dict:
        """
        Put together metadata using the output from the futures.
        Apply some systematics that necessitate the full sample to be processed.
        Apply normalizations for MC samples.
        Save the output.
        """

        self.logger.info("Postprocessing sample: " + sample)

        histograms = output.get("hists", {})
        cutflow = output.get("cutflow", {})
        gensumweight = output.get("gensumweight", value_accumulator(float, 0)).value

        metadata = {
            "ntuple_tag": self.options.tag,
            "analysis": self.options.channel,
            "isMC": int(self.options.isMC),
            "era": self.options.era,
            "sample": sample,
            "signal": (self.options.isMC) and (fill_utils.isSampleSignal(sample, self.options.era)),
            "xsec": 1,
            "gensumweight": gensumweight,
            "lumi": 1,
            "nfailed": output["_processing_metadata"].get("nfailed", value_accumulator(float, 0)).value,
            "nsuccess": output["_processing_metadata"].get("nsuccess", value_accumulator(float, 0)).value   
        }

        if self.options.isMC and self.options.doSyst:
            self.logger.info("Applying symmetric systematics.")

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
            self.logger.info("Applying normalization.")

            self.logger.debug(f"Found total_gensumweight {gensumweight}.")
            xsection = fill_utils.getXSection(sample, self.options.era, failOnKeyError=True)
            self.logger.debug(f"Found cross section x kr x br: {xsection}.")
            lumi = plot_utils.getLumi(self.options.era, scouting='scout' in self.options.channel)
            self.logger.debug(f"Found lumi: {lumi}.")

            if metadata['signal']:
                normalization = 1 / gensumweight
            else:
                normalization = xsection * lumi / gensumweight

            self.logger.info(f"Applying normalization: {normalization}.")
            histograms = fill_utils.apply_normalization(histograms, normalization)
            cutflow = fill_utils.apply_normalization(cutflow, normalization)

            # update metadata
            metadata['xsec'] = xsection
            metadata['lumi'] = lumi

        # update metadata
        if cutflow is not {}:
            for k, v in cutflow.items():
                metadata[k] = v

        # print the metadata before filling it with the git info
        self.logger.debug("Metadata:" + json.dumps(metadata, indent=4))

        commit, diff = fill_utils.get_git_info()
        metadata["git_commit"] = commit
        metadata["git_diff"] = diff

        self.write_output(sample, histograms, metadata)

    def setupSlurmClient(
            self,
            n_workers: int = 1, min_workers: int = 1, max_workers: int = 1,
            slurm_env: list = [], extra_args: list = []
        ) -> Client:

        # set default slurm environment variables
        if len(slurm_env) == 0:
            slurm_env = [
                'export DASK_DISTRIBUTED__COMM__ALLOWED_TRANSPORTS=["tcp://[::]:0"]',
                'export XRD_RUNFORKHANDLER=1',
                'export XRD_STREAMTIMEOUT=10',
                'echo "Landed on $HOSTNAME"',
                f'source {os.getenv("HOME")}/.bashrc',
                f'cd {os.chdir(os.path.dirname(os.path.abspath(__file__)))}',
                f"export PYTHONPATH=$PYTHONPATH:{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}",
                f'conda activate SUEP',
            ]

        # set default slurm extra arguments
        if len(extra_args) == 0:
            logDir = self.options.logDir
            logDir = os.path.join(logDir, "dask_histmaker", self.options.tag)
            if not os.path.exists(logDir):
                os.makedirs(logDir)
            extra_args=[
            f"--output={logDir}/job_output_%j.out",
            f"--error={logDir}/job_output_%j.err",
            "--partition=submit,submit-gpu",
            ]

        client = super().setupSlurmClient(n_workers, min_workers, max_workers, slurm_env, extra_args)

        return client

    def get_filelist(self, sample: str = '') -> List[str]:

        self.logger.debug(f"Getting list of files for sample {sample}.")
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

        self.logger.debug("Found {} files for sample {}.".format(len(files), sample))
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
            self.logger.info("Saving outputs to " + outFile)
            with open(outFile, "wb") as f:
                pickle.dump({"metadata": metadata, "hists": histograms}, f)
        else:
            outFile += ".root"
            self.logger.info("Saving outputs to " + outFile)
            with uproot.recreate(outFile) as froot:
                # write out metadata
                for k, m in metadata.items():
                    froot[f"metadata/{k}"] = str(m)

                # write out histograms
                for h, hist in histograms.items():
                    if len(hist.axes) > 3:
                        # TODO debug
                        self.logger.warning(
                           f"Skipping {h} because it has more than 3 axes. This is not supported by .root files."
                        )
                        continue
                    froot[h] = hist

    @staticmethod
    def process_file(ifile: str, sample: str, hists: dict, config: dict, options: SimpleNamespace) -> dict:
        """
        Read in ntuple hdf5 files and process each systematic variation to produce histograms and cutflows.
        Returns: a dictionary of histograms, cutflows, and gensumweight.
        """

        logging.debug(f"Processing file {ifile}.")

        output = {
            "hists": copy.deepcopy(hists),
            "cutflow": {},
            "gensumweight": value_accumulator(float, 0),
        }
        
        # get the file
        df, ntuple_metadata, ntuple_hists = fill_utils.open_ntuple(
            ifile, redirector=options.redirector, xrootd=options.xrootd
        )
        if options.printEvents:
            # in the case we are interested in the run, lumi, event numbers, we also need to know the file they're from
            print(f"Opened file {ifile}")

        # check if file is corrupted
        if type(df) == int:
            logging.debug(f"\tFile {ifile} is corrupted, skipping.")
            return output

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
            logging.debug(f"\tFound gensumweight {ntuple_metadata['gensumweight']} in ntuple.")
            output["gensumweight"] += ntuple_metadata["gensumweight"]

        # update the cutflows
        if ntuple_metadata != 0 and any(["cutflow" in k for k in ntuple_metadata.keys()]):
            logging.debug("\tFound cutflows in ntuple.")
            for k, v in ntuple_metadata.items():
                if "cutflow" in k:
                    if k not in output["cutflow"].keys():
                        output["cutflow"][k] = v
                    else:
                        output["cutflow"][k] += v

        # update the ntuple histograms
        for hist_name, hist in ntuple_hists.items():
            logging.debug(f"\tFound histograms {hist_name} in ntuple.")
            if hist_name in output["hists"]:
                output["hists"][hist_name] += hist
            else:
                output["hists"][hist_name] = hist

        # check if any events are in the ntuple dataframe
        if "empty" in list(df.keys()):
            logging.debug("\tNo events passed the selections, skipping.")
            return output
        if df.shape[0] == 0:
            logging.debug("\tNo events in file, skipping.")
            return output
        
        # we might modify this for systematics, so make a copy
        config = copy.deepcopy(config)

        for config_tag, config_out in config.items():

            logging.debug(f"\tUsing configuration {config_tag}.")

            logging.debug(f"\tRunning nominal variation")
            SUEPDaskHistMaker.plot_variation(df, '', options, config_out, config_tag, output["hists"], output["cutflow"], ntuple_metadata)

            if options.doSyst and options.isMC:

                for syst in config_out.get("syst", []):
                    logging.debug(f"Running syst {syst}")
                    SUEPDaskHistMaker.plot_variation(df, syst, options, config_out, config_tag, output["hists"], output["cutflow"], ntuple_metadata)

        # remove file at the end of loop
        if options.xrootd:
            fill_utils.close_ntuple(ifile)

        return output

    @staticmethod
    def plot_variation(
            df:pd.DataFrame, variation:str, options: SimpleNamespace, config_out:dict, config_tag:str,
            histograms:dict, cutflow:dict = {}, metadata:dict = {}
        ):

        df_plot = df.copy()

        # rename output method if we have applied a systematic
        if len(variation) > 0:
            config_tag = config_tag + "_" + variation

        logging.debug(f"\tFilling variation {variation} ({config_tag}).")

        # apply event weights
        df_plot = SUEPDaskHistMaker.apply_event_weights(df_plot, variation, options, metadata["sample"], config_tag)

        # prepare the DataFrame for plotting: blind, selections, new variables
        df_plot = fill_utils.prepare_DataFrame(
            df_plot,
            config_out,
            config_tag,
            isMC=options.isMC,
            blind=options.blind,
            cutflow=cutflow,
            output=histograms,
        )

        # if there are no events left after selections, no need to fill histograms
        if (df_plot is None) or (df_plot.shape[0] == 0):
            return histograms, cutflow, metadata

        # print out events that pass the selections, if requested
        if options.printEvents:
            print("Events passing selections for", config_tag)
            for index, row in df_plot.iterrows():
                print(
                    f"{int(row['event'])}, {int(row['run'])}, {int(row['luminosityBlock'])}"
                )

        # if no output histograms are defined for this method, skip it
        if config_tag and len([h for h in histograms.keys() if config_tag in h]) == 0:
            logging.warning(f"\tNo histograms defined for configuration {config_tag}, skipping filling histograms.")
            return histograms, cutflow, metadata

        # auto fill all histograms
        fill_utils.auto_fill(
            df_plot,
            histograms,
            config_out,
            config_tag,
            isMC=options.isMC,
            do_abcd=options.doABCD,
        )

        return histograms, cutflow, metadata

    @staticmethod
    def apply_event_weights(df: pd.DataFrame, variation: str, options: SimpleNamespace, sample: str = '', config_tag: str = '') -> pd.DataFrame:

        # TODO move this to fill_utils??

        # apply event weights
        if options.isMC:

            df["event_weight"] = df["genweight"].to_numpy()

            # 1) pileup weights
            puweights, puweights_up, puweights_down = pileup_weight.pileup_weight(
                options.era
            )
            pu = pileup_weight.get_pileup_weights(
                df, variation, puweights, puweights_up, puweights_down
            )
            df["event_weight"] *= pu

            # 2) PS weights
            if "PSWeight" in variation and variation in df.keys():
                df["event_weight"] *= df[variation]

            # 3) prefire weights
            if (options.era in ["2016apv", "2016", "2017"]):
                if "prefire" in variation and variation in df.keys():
                    df["event_weight"] *= df[variation]
                else:
                    df["event_weight"] *= df["prefire_nom"]

            # 4) TriggerSF weights
            if options.channel == "ggF":
                if options.scouting != 1:
                    (
                        trig_bins,
                        trig_weights,
                        trig_weights_up,
                        trig_weights_down,
                    ) = triggerSF.triggerSF(options.era)
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
                        np.array(df["ht"]).astype(int), variation, options.era
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
            if options.channel == 'WH' and options.isMC:
                if 'btag' in variation.lower():
                    btag_weights = variation
                else:
                    btag_weights = 'bTagWeight_nominal'
                btag_weights += "_" + config_tag
                if btag_weights not in df.keys():
                    logging.warning(f"btag weights {btag_weights} not found in DataFrame. Not applying them.")
                    pass
                else:
                    df['event_weight'] *= df[btag_weights]

        # data
        else:
            df["event_weight"] = np.ones(df.shape[0])

        return df

    
if __name__ == "__main__":

    # example usage

    options = {
        'isMC': 1,
        'channel': 'WH',
        'era': '2018',
        'tag': 'WH_7_20',
        'output': 'testDaskHistMaker',
        'doSyst': 0,
        'verbose': 0,
        'xrootd': 0,
        'saveDir': '/ceph/submit/data/user/l/lavezzo/SUEP/outputs/',
        'logDir': '/work/submit/lavezzo/SUEP/logs/',
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
    }

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
            ],
            "syst":  [
                "puweights_up",
                "puweights_down"
            ],
        },
    }
    hists = {}
    for output_method in config.keys():
        var_defs.initialize_new_variables(output_method, SimpleNamespace(**options), config[output_method])
        hist_defs.initialize_histograms(hists, output_method, SimpleNamespace(**options), config[output_method])
        if options.get("doSyst", False):
            for syst in config[output_method].get("syst", []):
                if any([j in syst for j in ["JER", "JES"]]):
                    config = fill_utils.get_jet_correction_config(config, syst)
    if options.get("doSyst", False): config = fill_utils.get_track_killing_config(config)
    
    
    histmaker = SUEPDaskHistMaker(config=config, options=options, hists=hists)
    client = histmaker.setupLocalClient(10)
    #client = histmaker.setupSlurmClient(n_workers=100, min_workers=2, max_workers=200)
    histmaker.run(client, [
        "WJetsToLNu_Pt-600ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM"
    ])
    client.close()