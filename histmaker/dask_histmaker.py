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

import logging
import os
import pickle
import subprocess
import sys
import uproot
import dask
import copy
import socket
import json
import numpy as np
import pandas as pd
from dask import delayed
from typing import Optional, List, Tuple
from types import SimpleNamespace
from dask.distributed import Client, LocalCluster, as_completed, Future
from dask_jobqueue import SLURMCluster
from coffea.processor.accumulator import AccumulatorABC
from coffea.processor import value_accumulator, dict_accumulator

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

# TODO debug
logging.basicConfig(level=logging.DEBUG)

class BaseDaskHistMaker():

    def __init__(self, **kwargs) -> None:

        # TODO debug
        # pass

        self.logger = kwargs.get("logger", logging.getLogger(self.__class__.__name__))
        self.logger.setLevel(logging.INFO)

    def setupLocalClient(self, n_workers: int = 1) -> Client:

        client = LocalCluster(
            n_workers=n_workers,
            dashboard_address='1776',
        ).get_client()
        self.logger.info(client)

        return client
    
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
            ]

        # set default slurm extra arguments
        if len(extra_args) == 0:
            logDir = "./logs_dask_histmaker/"
            if not os.path.exists(logDir):
                os.makedirs(logDir)
            extra_args=[
                f"--output={logDir}/job_output_%j.out",
                f"--error={logDir}/job_output_%j.err",
                "--partition=submit,submit-gpu",
            ]

        cluster = SLURMCluster(
            job_name="dask-histmaker",
            cores=1,
            memory='2GB',
            scheduler_options={
                'dashboard_address': '1776',
                'host': socket.gethostname()
            },
            silence_logs="error",
            job_extra_directives=extra_args,
            job_script_prologue=slurm_env
        )

        cluster.scale(n_workers)
        cluster.adapt(minimum=min_workers, maximum=max_workers)
        client = Client(cluster, heartbeat_interval='10s', timeout='10000s')

        self.logger.info(client)

        return client
    
    def run(
        self,
        client: Client,
        samples: List[str]
    ) -> dict:

        for sample in samples:
            self.preprocess_sample(sample)

        futures = {}
        for sample in samples:
            sample_futures = self.process_sample(client, sample)
            # this is used for internal tracking of which sample the output belongs to
            for future in sample_futures:
                future._sample = sample
            futures[sample] = sample_futures

        # flatten futures to a list to execute them
        futures = [future for sublist in futures.values() for future in sublist]

        output = {}
        for sample in samples:
            output[sample] = {}
            output[sample]["_processing_metadata"] = dict_accumulator({})
            output[sample]["_processing_metadata"]["nfailed"] = value_accumulator(float, 0)
            output[sample]["_processing_metadata"]["nsuccess"] = value_accumulator(float, 0)

        for future in as_completed(futures):

            sample = future._sample

            try:
                result = future.result()
                
                for key, value in result.items():

                    # coffea's accumulators can be added together
                    if issubclass(type(value), AccumulatorABC):
                
                        if key in output[sample].keys():
                            output[sample][key] += value
                        else:
                            
                            output[sample][key] = value

                    # we can also add dictionaries (e.g. of hist histograms, cutflows)
                    elif type(value) is dict:

                        if key in output[sample].keys():
                            for subkey in value.keys():
                                if subkey in output[sample][key].keys():
                                    output[sample][key][subkey] += value[subkey]
                                else:
                                    output[sample][key][subkey] = value[subkey]
                        else:
                            output[sample][key] = value

                    else:
                        raise Exception(f"Type {type(value)} not supported for output.")                

                output[sample]["_processing_metadata"]["nsuccess"] += 1

            except Exception as e:

                self.logger.error(f"Failed to merge output: {e}")
                output[sample]["_processing_metadata"]["nfailed"] += 1
                continue

        for sample in output.keys():
            output[sample] = self.postprocess_sample(sample, output[sample])

        return output
        
    def preprocess_sample(self, sample: str):
        """
        To be written by the user.
        """
        pass

    def process_sample(self, client, sample: str) -> List[Future]:
        """
        To be written by the user.
        Returns a list of dask futures to be executed.
        """
        pass

    def postprocess_sample(self, sample: str, output: dict) -> dict:
        """
        To be written by the user.
        Processes output from the futures and returns the final output.
        """
        pass
