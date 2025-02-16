"""
Author: Luca Lavezzo, Pietro Lugato
Date: August 2024
"""

import logging
import os
import gc
import socket
import traceback
import numbers
from time import time
from typing import List
from dask.distributed import Client, Future, LocalCluster, as_completed, progress
from dask_jobqueue import SLURMCluster
from dask import delayed
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


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

        self.logger.info(f"Setting up LocalClient with {n_workers} workers.")
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            dashboard_address="1776",
        )
        client = Client(cluster)
        self.logger.info(f"Workers ready. LocalClient ready.")
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
            walltime="5:00:00",
            memory="4GB",
            scheduler_options={
                "dashboard_address": "1776",
                "host": socket.gethostname(),
            },
            silence_logs="warning",
            job_extra_directives=extra_args,
            job_script_prologue=slurm_env,
        )

        cluster.scale(n_workers)
        # cluster.adapt(minimum=min_workers, maximum=max_workers) # this seems to make things very unstable with Slurm
        client = Client(cluster)

        self.logger.info(f"Waiting for workers to be ready... will start with {min_workers} workers, or timeout if not available within 2 minutes.")
        client.wait_for_workers(n_workers=min_workers, timeout=120)
        active_workers = len(client.scheduler_info()['workers'])
        self.logger.info(f"Workers ready. SLURMClient ready. Proceeding with {active_workers} workers. Will try to reach {n_workers} workers as more become avaiable.")
        self.logger.info(client)

        return client

    def print_ssh_command(self, client: Client) -> None:
        """
        Print the ssh command to connect to the dask dashboard
        to the screen for convenience.
        """

        dashboard_port = client.scheduler_info()['services']['dashboard']
        user = os.getenv("USER")
        hostname = os.getenv("HOSTNAME")
        ssh_command = f"ssh -L 8000:localhost:{dashboard_port} {user}@{hostname}"
        print("\nTo connect to the dask dashboard, run the following command in a separate terminal window:")
        print(ssh_command)
        print("And connect to http://localhost:8000 in your browser.\n")

    @staticmethod
    @delayed
    def merge(input1: dict, input2: dict) -> dict:
        """
        Merge two outputs from process_sample() efficiently.
        Returns the merged output.
        """
        for key, value in input2.items():
            # Handle dictionaries
            if isinstance(value, dict):
                if key not in input1:
                    input1[key] = {}
                for subkey, subvalue in value.items():
                    input1[key][subkey] = input1[key].get(subkey, 0) + subvalue
            # Handle numeric values
            elif isinstance(value, numbers.Number):
                input1[key] = input1.get(key, 0) + value
            # Raise an error for unsupported types
            else:
                raise TypeError(f"Type {type(value)} not supported for output.")
        return input1

    @staticmethod
    @delayed
    def _process(function, *args, **kwargs):
        """
        A generic function wrapper we use to submit to the client.
        """
        return function(*args, **kwargs)

    def build_metadata(self, samples: List[str]) -> dict:

        _run_metadata = {}
        _run_metadata["_processing_metadata"] = {}
        _run_metadata["_processing_metadata"]["t_start"] = time()

        for sample in samples:
            _run_metadata[sample] = {}
            _run_metadata[sample]["_processing_metadata"] = {}
            _run_metadata[sample]["_processing_metadata"]["postprocess_status"] = ""

        return _run_metadata
    
    def build_tree_graph(self, sample: str, processes: list, _run_metadata: dict) -> Future:
        """
        Construct the task graph for a given sample.
        Return the last node of the graph as a Delayed future.

        Seems to work better than build_triangle_graph.
        """

        self.logger.debug(f"Building graph for sample: {sample}")
        _run_metadata[sample]["_processing_metadata"]["n_total"] = len(processes)

        merged = []
        for i in range(0, len(processes), 2):
            if i+1 < len(processes):
                left = self._process(processes[i][0], *processes[i][1:])
                right = self._process(processes[i+1][0], *processes[i+1][1:])
                addition = self.merge(left, right)
            else:
                addition = self._process(processes[i][0], *processes[i][1:])
            merged.append(addition)

        while len(merged) > 1:
            new_merged = []
            for i in range(0, len(merged), 2):
                if i+1 < len(merged):
                    left = merged[i]
                    right = merged[i+1]
                    addition = self.merge(left, right)
                else:
                    addition = merged[i]
                new_merged.append(addition)
            merged = new_merged

        return merged[0]

    def get_batches(self, samples: List[str], _run_metadata: dict, batch_size: int = 1000) -> list:
        """
        From the size of each sample's graph, construct a list of batches of samples to be processed.
        Each batch is a list of samples
        """

        batches = []
        batch = []
        n_processing = 0
        for sample in samples:
            
            try:
                n_processing += _run_metadata[sample]["_processing_metadata"]["n_total"]
                batch.append(sample)

            except Exception as e:
                self.logger.error(f"Failed to process sample {sample}: {e}")
                self.logger.error(traceback.format_exc())
                continue

            if (n_processing >= batch_size) or (sample == samples[-1]):

                batches.append(batch)
                batch = []
                n_processing = 0

        self.logger.debug(f"Created {len(batches)} batches with sizes:")
        self.logger.debug([len(batch) for batch in batches])

        return batches

    def run(self, client: Client, samples: List[str], batch_size: int = 1000) -> dict:

        _run_metadata = self.build_metadata(samples)

        self.logger.info(f"Preprocessing samples.")
        for sample in samples:
            self.preprocess_sample(sample)
        _run_metadata["_processing_metadata"]["t_preprocess"] = time()

        self.logger.info(f"Creating graphs for samples.")
        graphs = {}
        for sample in samples:
            self.logger.debug(f"Processing sample: {sample}")
            processes = self.process_sample(sample)
            if len(processes) == 0:
                self.logger.error(f"Skipping {sample}.")
                samples.remove(sample)
                continue
            graphs[sample] = self.build_tree_graph(sample, processes, _run_metadata)

        self.logger.info(f"Creating batches.")
        batches = self.get_batches(samples, _run_metadata, batch_size=batch_size)
        n_batches = int(len(batches))
        _run_metadata["_processing_metadata"]["t_build"] = time()

        self.logger.info(f"Submitting futures to client in {n_batches} batch(es).")
        output = {sample: {} for sample in samples}
        with tqdm(total=n_batches, desc="Processing", position=0) as pbar:
            for i, batch in enumerate(batches):
                self.logger.debug(f"Processing batch {i+1}/{n_batches}.")

                try: 

                    # submit all samples in the batch
                    futures_in_progress = []
                    for sample in batch:
                        try:
                            sample_graph = graphs[sample]
                            futures = client.compute(sample_graph)
                            futures_in_progress.append(futures)
                        except Exception as e:
                            self.logger.error(f"Failed to process sample {sample}: {e}")
                            self.logger.error(traceback.format_exc())
                            continue

                    # collect futures, then delete them to free up worker memory
                    self.logger.debug(f"Collecting futures for batch {i+1}/{n_batches}.")
                    results = client.gather(futures_in_progress)
                    
                    # store output for the samples in the batch
                    for sample, result in dict(zip(batch, results)).items():
                        output[sample].update(result)

                except Exception as e:
                    self.logger.error(f"Failed to process batch {i+1}/{n_batches}: {e}")
                    self.logger.error(traceback.format_exc())
                    pbar.update(1)
                    continue

                pbar.update(1)
        _run_metadata["_processing_metadata"]["t_process"] = time()

        # add metadata to the output
        for sample in samples:
            output[sample].update(_run_metadata[sample])
        output["_processing_metadata"] = _run_metadata["_processing_metadata"]

        self.logger.info(f"Postprocessing samples.")
        for sample in samples:
            try:
                output[sample].update(self.postprocess_sample(sample, output[sample]))
                _run_metadata[sample]["_processing_metadata"]["postprocess_status"] = "success"
            except Exception as e:
                self.logger.error(f"Failed to postprocess sample {sample}: {e}")
                self.logger.error(traceback.format_exc())
                _run_metadata[sample]["_processing_metadata"]["postprocess_status"] = "failed"
                continue
        _run_metadata["_processing_metadata"]["t_postprocess"] = time()

        self.print_summary(samples, _run_metadata)

        return output

    def print_summary(self, samples: list, _run_metadata: dict) -> None:
        """
        Print a summary of the run from the metadata.
        """

        try:

            _tot_futures_results = {}

            self.logger.info("")
            self.logger.info("Run Summary:")

            self.logger.debug("")
            for sample in samples:
                self.logger.debug(f"Sample: {sample}")
                for key, value in _run_metadata[sample]["_processing_metadata"].items():
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
                            if _run_metadata[s]["_processing_metadata"]["postprocess_status"]
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
                            if _run_metadata[s]["_processing_metadata"]["postprocess_status"]
                            == "failed"
                        ]
                    )
                )
            )

            self.logger.info("")
            self.logger.info(
                f"Time to preprocess: {_run_metadata['_processing_metadata']['t_preprocess'] - _run_metadata['_processing_metadata']['t_start']:.2f} s"
            )
            self.logger.info(
                f"Time to build graphs: {_run_metadata['_processing_metadata']['t_build'] - _run_metadata['_processing_metadata']['t_preprocess']:.2f} s"
            )
            self.logger.info(
                f"Time to process: {_run_metadata['_processing_metadata']['t_process'] - _run_metadata['_processing_metadata']['t_build']:.2f} s"
            )
            self.logger.info(
                f"Time to postprocess: {_run_metadata['_processing_metadata']['t_postprocess'] - _run_metadata['_processing_metadata']['t_process']:.2f} s"
            )
            self.logger.info(
                f"Total time: {_run_metadata['_processing_metadata']['t_postprocess'] - _run_metadata['_processing_metadata']['t_start']:.2f} s"
            )

        except Exception as e:
            self.logger.error(f"Failed to print summary: {e}")
            self.logger.error(traceback.format_exc())

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
