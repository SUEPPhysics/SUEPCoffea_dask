"""
Author: Luca Lavezzo
Date: August 2024
"""

import logging
import os
import socket
from typing import List
from dask.distributed import Client, LocalCluster, as_completed, Future
from dask_jobqueue import SLURMCluster
from coffea.processor.accumulator import AccumulatorABC
from coffea.processor import value_accumulator, dict_accumulator


class BaseDaskHistMaker():
    """
    A base class for making histograms with dask, a la coffea processor.
    The user is expected to write preprocess_sample, process_sample, and postprocess_sample methods.
    :preprocess_sample: does not return anything, but is used to prepare the sample for processing.
    :process_sample: returns a list of dask futures to be executed. Each of which generates an output that is merged together: coffea processors or dictionaries of objects that can be added (floats, histograms, etc.)
    :postprocess_sample: processes the output from the futures and returns the final output.
    """

    def __init__(self, **kwargs) -> None:

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
            output[sample]["_processing_metadata"]["postprocess_status"] = ""
            output[sample]["_processing_metadata"]["n_processed"] = value_accumulator(float, 0)
            output[sample]["_processing_metadata"]["n_success"] = value_accumulator(float, 0)
            output[sample]["_processing_metadata"]["n_failed"] = value_accumulator(float, 0)

        for future in as_completed(futures):

            output[sample]["_processing_metadata"]["n_processed"] += 1

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

                output[sample]["_processing_metadata"]["n_success"] += 1

            except Exception as e:

                self.logger.error(f"Failed to merge output: {e}")
                output[sample]["_processing_metadata"]["n_failed"] += 1
                continue

        for sample in list(output.keys()):
            try:
                output[sample].update(self.postprocess_sample(sample, output[sample]))
                output[sample]["_processing_metadata"]["postprocess_status"] = "success"
            except Exception as e:
                self.logger.error(f"Failed to postprocess sample {sample}: {e}")
                output[sample]["_processing_metadata"]["postprocess_status"] = "failed"
                continue

        self.print_summary(output)

        return output
    
    def print_summary(self, output: dict) -> None:
        
        _tot_futures_results = {}

        self.logger.info("Run Summary:")
        for sample in output.keys():
            self.logger.info(f"Sample: {sample}")
            self.logger.info(f"\tStatus: {output[sample]['_processing_metadata']['status']}")
            for key, value in output[sample]['_processing_metadata'].items():
                if key.startswith("n_"):
                    status = key.split("_")[1]
                    self.logger.info(f"\t{status}: {value.value}")
                    if status not in _tot_futures_results.keys():
                        _tot_futures_results[status] = 0
                    _tot_futures_results[status] += value.value

        self.logger.info("")
        
        for status, value in _tot_futures_results.items():
            self.logger.info(f"Total futures {status}: {value}")

        self.logger.info("")

        self.logger.info(f"Total samples post-processed: {len(output.keys())}")
        self.logger.info("\tSamples Success: " + str(len([s for s in output.keys() if output[s]['_processing_metadata']['postprocess_status'] == 'success'])))
        self.logger.info("\tSamples Failed: " + str(len([s for s in output.keys() if output[s]['_processing_metadata']['postprocess_status'] == 'failed'])))
        
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


