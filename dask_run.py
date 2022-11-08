import os, sys, glob
import pandas as pd
import json
import argparse
import time
import numpy
from tqdm import tqdm

from dask.distributed import Client, Worker, WorkerPlugin
from typing import List
import shutil

# Import coffea specific features
from coffea import processor, nanoevents

# SUEP Repo Specific
import SUEP_coffea
import genSumWeightExtractor
from workflows import pandas_utils
from workflows import merger

class DependencyInstaller(WorkerPlugin):
    def __init__(self, dependencies: List[str]):
        self._depencendies = " ".join(f"'{dep}'" for dep in dependencies)

    def setup(self, worker: Worker):
        os.system(f"pip install {self._depencendies}")

dependency_installer = DependencyInstaller([
    "fastjet --upgrade",
])

client = Client("tls://<some-address-to-get>:<port>")
client.register_worker_plugin(dependency_installer)
shutil.make_archive("workflows", "zip", base_dir="workflows")

def main():
    processor_inst = SUEP_coffea.SUEP_cluster(
        isMC=1,
        era=2018,
        do_syst=1,
        syst_var="",
        sample="QCD_Pt+RunIISummer20UL18",
        weight_syst="",
        flag=False,
        output_location='.'
    )

    result = processor.run_uproot_job(
        fileset="filelist/json_lpcsuep/QCD_Pt_full_xcache.json",
        treename="Events",
        processor_instance=processor_inst,
        executor=processor.dask_executor,
        executor_args={
            "schema": processor.NanoAODSchema,
            "client": client,
        },
        chunksize=5000,
    )

if __name__ == "__main__":
    main()

