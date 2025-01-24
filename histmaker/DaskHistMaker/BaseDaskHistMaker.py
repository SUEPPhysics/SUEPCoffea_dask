"""
Author: Luca Lavezzo
Date: August 2024
"""

import logging
import os
import socket
import traceback
import numbers
from time import time
from typing import List
from dask.distributed import Client, Future, LocalCluster, as_completed
from dask_jobqueue import SLURMCluster
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

        cluster = LocalCluster(
            threads_per_worker=1,
            dashboard_address="1776",
        )
        cluster.scale(n_workers)
        client = Client(cluster)
        self.logger.info(f"Waiting for workers to be ready...")
        client.wait_for_workers(1)
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

        self.logger.info(f"Waiting for workers to be ready...")
        client.wait_for_workers(1)
        self.logger.info(f"Workers ready. SLURMClient ready.")
        self.logger.info(client)

        return client

    @staticmethod
    def merge_samples(input1: dict, input2: dict) -> dict:
        """
        Merge two samples efficiently.
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
    def merge_futures(input1: dict, input2: dict) -> dict:
        """
        Merges the output of two futures efficiently.
        Returns the merged output.
        """
        try:
            # Ensure we always iterate over the smaller dictionary
            if len(input1) > len(input2):
                input1, input2 = input2, input1

            # Merge dictionaries
            for sample, value in input1.items():
                if sample in input2:
                    input2[sample] = BaseDaskHistMaker.merge_samples(input2[sample], value)
                else:
                    input2[sample] = value

            return input2

        except Exception as e:
            print(f"Failed to merge output: {e}")
            print(traceback.format_exc())
            return {}


    def run(self, client: Client, samples: List[str]) -> dict:

        _metadata = {}
        _metadata["_processing_metadata"] = {}

        _metadata["_processing_metadata"]["t_start"] = time()

        self.logger.info(f"Preprocessing samples.")
        for sample in samples:
            self.preprocess_sample(sample)

        _metadata["_processing_metadata"]["t_preprocess"] = time()

        self.logger.info(f"Processing samples.")
        futures_dict = {}
        for sample in samples:
            sample_futures = self.process_sample(client, sample)
            # this is used for internal tracking of which sample the output belongs to
            for future in sample_futures:
                future._sample = sample
            futures_dict[sample] = sample_futures

        # flatten futures to a list to execute them
        futures = [future for sublist in futures_dict.values() for future in sublist]

        # add some sample metadata to the output
        for sample in samples:
            _metadata[sample] = {}
            _metadata[sample]["_processing_metadata"] = {}
            _metadata[sample]["_processing_metadata"]["postprocess_status"] = ""
            _metadata[sample]["_processing_metadata"]["n_processed"] = 0
            _metadata[sample]["_processing_metadata"]["n_success"] = 0
            _metadata[sample]["_processing_metadata"]["n_failed"] = 0

        # multi-threaded merging of future results
        # (this avoids the possible issue of workers crashing due to memory overload by
        # freeing up their memory more quickly, and generally speeds up the process)
        self.logger.info(f"Processing and collecting futures.")
        # with ThreadPoolExecutor(max_workers=10) as executor:
        #     reading_futures = []
        #     for future in tqdm(as_completed(futures), total=len(futures)):
        #         reading_futures.append(executor.submit(self.merge_future, future, output, self.logger))
        #     for future in reading_futures:
        #         try:
        #             if future.result(timeout=60) == 1:
        #                 self.logger.error(f"Failed to process future.")
        #         except TimeoutError:
        #             self.logger.error(f"TimeoutError: Future took too long to complete.")
        #             output[future._sample]["_processing_metadata"]["n_failed"] += 1
        #             continue  
        # 

        sequence = as_completed(futures)   

        def grab_next_result(sequence):

            if sequence.count() == 0:
                return None

            try:

                future = next(sequence)
                result = future.result()
                future.release()

                # ugly workaround should fix elsewhere
                if 'merge_futures' not in future.key:
                    sample = future._sample
                    result = {sample: result}
                    future_type = 'process'
                else:
                    future_type = 'merge'

                return result, future_type

            except Exception as e:
                
                print(f"Failed to grab result: {e}")
                print(traceback.format_exc())
                return grab_next_result(sequence) 

        def update_pbar(future_type):
            
            if future_type == 'process':
                # +1 processed, +1 total that need to be merged
                process_pbar.update(1)
                merge_pbar.reset(total=merge_pbar.total + 1)
            elif future_type == 'merge':
                # +1 merged
                merge_pbar.update(1)

        process_pbar = tqdm(total=sequence.count(), desc="Processing", position=0)
        merge_pbar = tqdm(total=0, desc="Merging", position=1)
            
        while sequence.count() > 1:

            result1, future_type1 = grab_next_result(sequence)
            update_pbar(future_type1)
            result2, future_type2 = grab_next_result(sequence)
            update_pbar(future_type2)

            new = client.submit(
                self.merge_futures,
                result1,
                result2,
                priority=1000,
            )

            sequence.add(new)

        process_pbar.close()
        merge_pbar.close()
            
        output = next(sequence).result()
        for sample in samples:
            output[sample].update(_metadata[sample])
        output["_processing_metadata"] = _metadata["_processing_metadata"]

        output["_processing_metadata"]["t_process"] = time()

        self.logger.info(f"Postprocessing samples.")
        for sample in samples:
            try:
                output[sample].update(self.postprocess_sample(sample, output[sample]))
                output[sample]["_processing_metadata"]["postprocess_status"] = "success"
            except Exception as e:
                self.logger.error(f"Failed to postprocess sample {sample}: {e}")
                self.logger.error(traceback.format_exc())
                output[sample]["_processing_metadata"]["postprocess_status"] = "failed"
                continue

        output["_processing_metadata"]["t_postprocess"] = time()

        self.print_summary(_metadata, samples)

        return output

    def print_summary(self, metadata: dict, samples: list) -> None:

        _tot_futures_results = {}

        self.logger.info("")
        self.logger.info("Run Summary:")

        self.logger.debug("")
        for sample in samples:
            self.logger.debug(f"Sample: {sample}")
            for key, value in metadata[sample]["_processing_metadata"].items():
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
                        if metadata[s]["_processing_metadata"]["postprocess_status"]
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
                        if metadata[s]["_processing_metadata"]["postprocess_status"]
                        == "failed"
                    ]
                )
            )
        )

        self.logger.info("")
        self.logger.info(
            f"Time to preprocess: {metadata['_processing_metadata']['t_preprocess'] - metadata['_processing_metadata']['t_start']:.2f} s"
        )
        self.logger.info(
            f"Time to process: {metadata['_processing_metadata']['t_process'] - metadata['_processing_metadata']['t_preprocess']:.2f} s"
        )
        self.logger.info(
            f"Time to postprocess: {metadata['_processing_metadata']['t_postprocess'] - metadata['_processing_metadata']['t_process']:.2f} s"
        )
        self.logger.info(
            f"Total time: {metadata['_processing_metadata']['t_postprocess'] - metadata['_processing_metadata']['t_start']:.2f} s"
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
