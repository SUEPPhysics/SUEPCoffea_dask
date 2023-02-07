import argparse
import logging
import os
import subprocess
import time

from plotting.plot_utils import check_proxy

logging.basicConfig(level=logging.DEBUG)


def cleanCorruptedFiles(out_dir_sample):
    for file in os.listdir(out_dir_sample):
        size = os.path.getsize(out_dir_sample + "/" + file)
        if size == 0:
            subprocess.run(["rm", out_dir_sample + "/" + file])


def main():
    parser = argparse.ArgumentParser(description="Famous Submitter")
    parser.add_argument("-t", "--tag", type=str, help="Dataset tag.")
    parser.add_argument(
        "-r",
        "--resubmits",
        type=int,
        default=10,
        help="Number of resubmissions.",
        required=False,
    )
    parser.add_argument(
        "-hours",
        "--hours",
        type=float,
        default=1.0,
        help="Number of hours per resubmission, in addition to the time between sample submissions.",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data.txt",
        help="input datasets",
        required=True,
    )
    parser.add_argument(
        "-dry", "--dryrun", type=int, default=0, help="running without submission"
    )

    options = parser.parse_args()
    nResubmits = options.resubmits
    nHours = options.hours
    tag = options.tag

    username = os.environ["USER"]
    dataDir = f"/mnt/T3_US_MIT/hadoop/scratch/{username}/SUEP/{tag}/"

    # Making sure that the proxy is good
    lifetime = check_proxy(time_min=200)
    logging.info(f"--- proxy lifetime is {round(lifetime, 1)} hours")

    logging.info("Running resubmission script from " + str(os.environ["HOSTNAME"]))
    for i in range(nResubmits):
        logging.info("Resubmission " + str(i))
        logging.info("Removing all jobs...")
        os.system("condor_rm {}".format(os.environ["USER"]))
        logging.info("Checking for corrupted files and removing them...")

        t_start = time.time()

        # delete files that are corrupted
        subDirs = os.listdir(dataDir)
        for subDir in subDirs:
            cleanCorruptedFiles(dataDir + "/" + subDir)

        if not options.dryrun:
            logging.info("Executing monitor.py...")
            os.system(
                "python3 monitor.py --tag={} --input={} -r=1 -m={}".format(
                    tag, options.input, options.movesample
                )
            )

        t_end = time.time()

        # don't wait if it's the last submission
        if i == nResubmits - 1:
            logging.info("All done")
            break

        # wait to resubmit jobs using the parameter <hours>, accounts for time it took to submit them
        sleepTime = 60 * 60 * nHours
        mod = t_end - t_start
        logging.info("Submitting and moving files took " + str(round(mod)) + " seconds")
        if sleepTime - mod <= 0:
            continue
        if nHours > 0:
            logging.info("Sleeping for " + str(round(sleepTime - mod)) + " seconds")
            logging.info("(" + str(round(nHours - mod * 1.0 / 3600, 2)) + " hours)...")
        time.sleep(sleepTime - mod)


if __name__ == "__main__":
    main()
