import argparse
import logging
import os
import shutil
import subprocess
import time

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Famous Submitter")
parser.add_argument("-t", "--tag", type=str, help="Dataset tag.")
parser.add_argument("-i", "--input", type=str, help="Input filelist")
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
    "-ms",
    "--movesample",
    type=int,
    default=0,
    help="Move each sample after submitting it (accomplishes it during the buffer time between samples set by default in monitor.py).",
)
parser.add_argument(
    "-m",
    "--move",
    type=int,
    default=0,
    help="Move all samples after all submissions (during the buffer specified by -hours).",
)
parser.add_argument(
    "-dry", "--dryrun", type=int, default=0, help="running without submission"
)

options = parser.parse_args()
nResubmits = options.resubmits
nHours = options.hours
tag = options.tag

username = os.environ["USER"]
dataDir = f"/data/submit/{username}/SUEP/{tag}/"
moveDir = f"/work/submit/{username}/SUEP/{tag}/"

# Making sure that the proxy is good
proxy_base = f"x509up_u{os.getuid()}"
home_base = os.environ["HOME"]
proxy_copy = os.path.join(home_base, proxy_base)
regenerate_proxy = False
if not os.path.isfile(proxy_copy):
    logging.warning("--- proxy file does not exist")
    regenerate_proxy = True
else:
    lifetime = subprocess.check_output(
        ["voms-proxy-info", "--file", proxy_copy, "--timeleft"]
    )
    lifetime = float(lifetime)
    lifetime = lifetime / (60 * 60)
    logging.info(f"--- proxy lifetime is {round(lifetime, 1)} hours")
    if lifetime < nHours * nResubmits * 1.5:
        logging.warning("--- proxy has expired !")
        regenerate_proxy = True

if regenerate_proxy:
    redone_proxy = False
    while not redone_proxy:
        status = os.system(
            "voms-proxy-init -voms cms --hours=" + str(nHours * nResubmits * 1.5)
        )
        if os.WEXITSTATUS(status) == 0:
            redone_proxy = True
    shutil.copyfile("/tmp/" + proxy_base, proxy_copy)


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
        for file in os.listdir(dataDir + subDir):
            size = os.path.getsize(dataDir + subDir + "/" + file)
            if size == 0:
                subprocess.run(["rm", dataDir + subDir + "/" + file])
            elif size < 5000:
                subprocess.run(["rm", dataDir + subDir + "/" + file])

    if not options.dryrun:
        logging.info("Executing monitor.py ...")
        os.system(
            "python3 monitor.py --tag={} --input={} -r=1 -m={}".format(
                tag, options.input, options.movesample
            )
        )

    if options.move:
        if not os.path.isdir(moveDir):
            os.system("mkdir " + moveDir)

        subDirs = os.listdir(dataDir)

        for subDir in subDirs:
            if not os.path.isdir(moveDir + subDir):
                os.system("mkdir " + moveDir + subDir)

            # get list of files already in /work
            movedFiles = os.listdir(moveDir + subDir)

            # get list of files in T3
            allFiles = os.listdir(dataDir + subDir)

            # get list of files missing from /work that are in T3
            filesToMove = list(set(allFiles) - set(movedFiles))

            # move those files
            logging.info(
                "Moving " + str(len(filesToMove)) + " files to " + moveDir + subDir
            )
            for file in filesToMove:
                subprocess.run(
                    [
                        "xrdcp",
                        "root://submit50.mit.edu/" + dataDir + subDir + "/" + file,
                        moveDir + subDir + "/",
                    ]
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
