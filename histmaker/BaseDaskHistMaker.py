"""
Author: Luca Lavezzo
Date: August 2024
"""

import logging
import os
import socket
import traceback
from time import time
from typing import List
from dask.distributed import Client, Future, LocalCluster, as_completed
from dask_jobqueue import SLURMCluster
from tqdm import tqdm


class BaseDaskHistMaker:
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
            dashboard_address="1776",
        ).get_client()
        self.logger.info(client)

        return client

    def setupSlurmClient(
        self,
        n_workers: int = 1,
        min_workers: int = 1,
        max_workers: int = 1,
        slurm_env: list = [],
        extra_args: list = [],
    ) -> Client:

        # set default slurm environment variables
        if len(slurm_env) == 0:
            slurm_env = [
                'export DASK_DISTRIBUTED__COMM__ALLOWED_TRANSPORTS=["tcp://[::]:0"]',
                "export XRD_RUNFORKHANDLER=1",
                "export XRD_STREAMTIMEOUT=10",
                'echo "Landed on $HOSTNAME"',
                f'source {os.getenv("HOME")}/.bashrc',
                f"cd {os.chdir(os.path.dirname(os.path.abspath(__file__)))}",
                f"export PYTHONPATH=$PYTHONPATH:{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}",
            ]

        # set default slurm extra arguments
        if len(extra_args) == 0:
            logDir = "./logs_dask_histmaker/"
            if not os.path.exists(logDir):
                os.makedirs(logDir)
            extra_args = [
                f"--output={logDir}/job_output_%j.out",
                f"--error={logDir}/job_output_%j.err",
                "--partition=submit,submit-gpu",
            ]

        cluster = SLURMCluster(
            job_name="dask-histmaker",
            cores=1,
            memory="4GB",
            scheduler_options={
                "dashboard_address": "1776",
                "host": socket.gethostname(),
            },
            silence_logs="error",
            job_extra_directives=extra_args,
            job_script_prologue=slurm_env,
        )

        cluster.scale(n_workers)
        # cluster.adapt(minimum=min_workers, maximum=max_workers) # this seems to make things very unstable with Slurm
        client = Client(cluster)

        self.logger.info(f"Waiting for workers to be ready...")
        client.wait_for_workers(1)
        self.logger.info(f"Workers ready. SLURMClient ready.")
        self.logger.info(client)

        return client

    def run(self, client: Client, samples: List[str]) -> dict:

        output = {}
        output["_processing_metadata"] = {}

        output["_processing_metadata"]["t_start"] = time()

        self.logger.info(f"Preprocessing samples.")
        for sample in samples:
            self.preprocess_sample(sample)

        output["_processing_metadata"]["t_preprocess"] = time()

        self.logger.info(f"Processing samples.")
        futures = {}
        for sample in samples:
            sample_futures = self.process_sample(client, sample)
            # this is used for internal tracking of which sample the output belongs to
            for future in sample_futures:
                future._sample = sample
            futures[sample] = sample_futures

        # flatten futures to a list to execute them
        futures = [future for sublist in futures.values() for future in sublist]

        # add some sample metadata to the output
        for sample in samples:
            output[sample] = {}
            output[sample]["_processing_metadata"] = {}
            output[sample]["_processing_metadata"]["postprocess_status"] = ""
            output[sample]["_processing_metadata"]["n_processed"] = 0
            output[sample]["_processing_metadata"]["n_success"] = 0
            output[sample]["_processing_metadata"]["n_failed"] = 0

        self.logger.info(f"Processing and collecting futures.")
        for future in tqdm(as_completed(futures), total=len(futures)):

            sample = future._sample

            output[sample]["_processing_metadata"]["n_processed"] += 1

            try:
                result = future.result()

                for key, value in result.items():

                    # we can add dictionaries (e.g. of hist histograms, cutflows)
                    if type(value) is dict:

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
                self.logger.error(traceback.format_exc())
                output[sample]["_processing_metadata"]["n_failed"] += 1
                continue

        output["_processing_metadata"]["t_process"] = time()

        self.logger.info(f"Postprocessing samples.")
        for sample in samples:
            try:
                output[sample].update(self.postprocess_sample(sample, output[sample]))
                output[sample]["_processing_metadata"]["postprocess_status"] = "success"
            except Exception as e:
                self.logger.error(f"Failed to postprocess sample {sample}: {e}")
                output[sample]["_processing_metadata"]["postprocess_status"] = "failed"
                continue

        output["_processing_metadata"]["t_postprocess"] = time()

        self.print_summary(output, samples)

        return output

    def print_summary(self, output: dict, samples: list) -> None:

        _tot_futures_results = {}

        self.logger.info("")
        self.logger.info("Run Summary:")

        self.logger.debug("")
        for sample in samples:
            self.logger.debug(f"Sample: {sample}")
            for key, value in output[sample]["_processing_metadata"].items():
                if key.startswith("n_"):
                    status = key.split("_")[1]
                    self.logger.debug(f" {status}: {value}")
                    if status not in _tot_futures_results.keys():
                        _tot_futures_results[status] = 0
                    _tot_futures_results[status] += value

        self.logger.info("")
        for status, value in _tot_futures_results.items():
            self.logger.info(f"Total futures {status}: {value}")

        self.logger.info("")
        self.logger.info(f"Total samples post-processed: {len(samples)}")
        self.logger.info(
            "\tSamples succeeded: "
            + str(
                len(
                    [
                        s
                        for s in samples
                        if output[s]["_processing_metadata"]["postprocess_status"]
                        == "success"
                    ]
                )
            )
        )
        self.logger.info(
            "\tSamples failed: "
            + str(
                len(
                    [
                        s
                        for s in samples
                        if output[s]["_processing_metadata"]["postprocess_status"]
                        == "failed"
                    ]
                )
            )
        )

        self.logger.info("")
        self.logger.info(
            f"Time to preprocess: {output['_processing_metadata']['t_preprocess'] - output['_processing_metadata']['t_start']:.2f} s"
        )
        self.logger.info(
            f"Time to process: {output['_processing_metadata']['t_process'] - output['_processing_metadata']['t_preprocess']:.2f} s"
        )
        self.logger.info(
            f"Time to postprocess: {output['_processing_metadata']['t_postprocess'] - output['_processing_metadata']['t_process']:.2f} s"
        )
        self.logger.info(
            f"Total time: {output['_processing_metadata']['t_postprocess'] - output['_processing_metadata']['t_start']:.2f} s"
        )

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
