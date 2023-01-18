import argparse
import logging
import os
import subprocess
from shutil import copyfile

from termcolor import colored

from plotting.plot_utils import check_proxy

logging.basicConfig(level=logging.DEBUG)


def cleanCorruptedFiles(out_dir_sample):
    for file in os.listdir(out_dir_sample):
        size = os.path.getsize(out_dir_sample + "/" + file)
        if size == 0:
            subprocess.run(["rm", out_dir_sample + "/" + file])


def main():
    parser = argparse.ArgumentParser(description="Famous Submitter")
    parser.add_argument("-i", "--input", type=str, default="input", required=True)
    parser.add_argument("-t", "--tag", type=str, default="IronMan", required=True)
    parser.add_argument("-r", "--resubmit", type=int, default=0, help="")
    options = parser.parse_args()

    username = os.environ["USER"]
    out_dir = (
        "/data/submit/cms/store/user/" + username + "/SUEP/" + options.tag + "/{}/"
    )
    jobs_dir = "/work/submit/" + username + "/SUEP/logs/"

    # Making sure that the proxy is good
    if options.resubmit:
        lifetime = check_proxy(time_min=100)
        logging.info(f"--- proxy lifetime is {round(lifetime, 1)} hours")

    with open(options.input) as stream:
        for sample in stream.read().split("\n"):
            if len(sample) <= 1 or "#" in sample:
                continue

            if "/" in sample and len(sample.split("/")) <= 1:
                continue

            if "/" in sample:
                sample_name = sample.split("/")[-1]
            else:
                sample_name = sample

            jobs_dir_sample = "_".join(["jobs", options.tag, sample_name])
            jobs_dir_sample = jobs_dir + jobs_dir_sample
            out_dir_sample = out_dir.format(sample_name)

            # delete files that are corrupted (i.e., empty)
            cleanCorruptedFiles(out_dir_sample)
            logging.info(jobs_dir_sample)

            # We write the original list. inputfiles.dat will now contain missing files. Compare with original list
            if not os.path.isfile(jobs_dir_sample + "/" + "original_inputfiles.dat"):
                copyfile(
                    jobs_dir_sample + "/" + "inputfiles.dat",
                    jobs_dir_sample + "/" + "original_inputfiles.dat",
                )

            # Find out the jobs that run vs the ones that failed
            jobs = [
                line.rstrip()
                for line in open(jobs_dir_sample + "/" + "original_inputfiles.dat")
            ]

            njobs = len(jobs)
            complete_list = os.listdir(out_dir_sample)
            nfile = len(complete_list)

            # Print out the results
            logging.info(
                "-- {:62s}".format(
                    (sample_name[:60] + "..") if len(sample_name) > 60 else sample_name
                )
            )

            # Print out the results
            percent = nfile / njobs * 100
            logging.info(
                colored("\t\t --> completed", "green")
                if njobs == nfile
                else colored(
                    "\t\t --> ({}/{}) finished. {:.1f}% complete".format(
                        nfile, njobs, percent
                    ),
                    "red",
                )
            )

            # If files are missing we resubmit with the same condor.sub
            if options.resubmit:
                logging.info(f"-- resubmitting files for {sample}")
                file_names = []
                for item in complete_list:
                    if "." not in item:
                        continue
                    file_names.append(os.path.splitext(item)[0])

                jobs_resubmit = [
                    item for item in jobs if item.split("\t")[-1] not in file_names
                ]
                resubmit_file = open(jobs_dir_sample + "/" + "inputfiles.dat", "w")
                for redo_file in jobs_resubmit:
                    resubmit_file.write(redo_file + "\n")
                resubmit_file.close()

                htc = subprocess.Popen(
                    "condor_submit " + os.path.join(jobs_dir_sample, "condor.sub"),
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    close_fds=True,
                )
                out, err = htc.communicate()
                exit_status = htc.returncode
                logging.info(f"condor submission status : {exit_status}")


if __name__ == "__main__":
    main()
