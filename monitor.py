import argparse
import logging
import os
import subprocess
import time
from shutil import copyfile
import numpy as np
import pandas as pd
from termcolor import colored
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


def isFileGood(fname, label="ch"):
    try:
        with pd.HDFStore(fname) as store:
            data = store[label]
        return 1
    except:
        return 0

def auto_mode(options):

    logging.basicConfig(level=logging.ERROR)
    completion_data = {}
    start_time = time.time()
    max_runtime = 7 * 24 * 60 * 60  # 1 week in seconds
    
    while True:
        # Check if the runtime has exceeded the maximum allowed time
        if time.time() - start_time > max_runtime:
            logging.info("Maximum runtime of 1 week exceeded. Exiting auto_mode.")
            break

        samples_data = monitor(options) 

        # Capture the current time and the completion rates
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        completion_data[current_time] = samples_data

        # Update plot
        plot_completion(options, completion_data)

        # Wait for the specified interval before next check
        logging.info(f"Sleeping for {options.auto} minutes...")
        time.sleep(options.auto * 60)


def plot_completion(options, completion_data):
    """
    Plot the completion progress over time.
    """
    times = list(completion_data.keys())
    formatted_times = pd.to_datetime(list(completion_data.keys()), format="%Y-%m-%d %H:%M:%S")
    
    total_completion = [
        sum([sample.get('completed', 0) for sample in completion_data[t].values()]) * 100 / sum([sample.get('total', 1) for sample in completion_data[t].values()]) for t in times
    ]
    
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    
    # Plot individual sample completion rates
    sample_names = list(next(iter(completion_data.values())).keys())
    
    for sample in sample_names:
        sample_completion = [
            completion_data[t][sample].get('completed', 0) * 100 / completion_data[t][sample].get('total', 1) for t in times
        ]
        ax.plot(formatted_times, sample_completion, label=sample[:30] + f" ({round(sample_completion[-1], 2)}%)")

    ax.plot(formatted_times, total_completion, label=f"Total ({round(total_completion[-1], 2)}%)", linewidth=3, color='black')

    # Format the x-axis to show date and time
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%m\n%H:%M'))

    # Plot settings
    # Set xticks to be at most 10 in number
    if len(formatted_times) > 10:
        xticks = [formatted_times[i] for i in np.linspace(0, len(formatted_times) - 1, 10, dtype=int)]
    else:
        xticks = formatted_times
    ax.set_xticks(xticks)
    ax.set_xlabel("Time")
    ax.set_ylabel("Percent Completed")
    ax.set_title("NTuple Tag: " + options.tag)
    ax.legend(loc=(1.01, 0), fontsize='small')
    fig.tight_layout()
    fig.savefig("/home/submit/lavezzo/public_html/monitoring/" + options.tag + ".pdf", bbox_inches="tight")

def main():

    parser = argparse.ArgumentParser(description="Famous Submitter")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="input",
        help="Input filelist.",
        required=True,
    )
    parser.add_argument("-t", "--tag", type=str, default="IronMan", required=True)
    parser.add_argument("-r", "--resubmit", type=int, default=0, help="")
    parser.add_argument(
        "-w",
        "--wait",
        type=float,
        default=0,
        help="Number of hours to wait between sample resubmissions.",
    )
    parser.add_argument(
        "-m",
        "--move",
        type=int,
        default=0,
        help="Move files to move_dir from out_dir_xrd while you check if they are corrupted.",
    )
    parser.add_argument(
        "--dataDirLocal",
        type=str,
        default="/ceph/submit/data/user/"
        + os.environ["USER"][0]
        + "/"
        + os.environ["USER"]
        + "/SUEP/",
    )
    parser.add_argument("-redirector", type=str, default="root://submit50.mit.edu/")
    parser.add_argument(
        "--auto",
        type=int,
        help="Automatically rerun every N minutes and plot completion rates."
    )
    options = parser.parse_args()

    if options.auto:
        auto_mode(options)
    else:
        monitor(options)

def monitor(options):

    data = {} # this is the output

    proxy_base = f"x509up_u{os.getuid()}"
    home_base = os.environ["HOME"]
    username = os.environ["USER"]
    proxy_copy = os.path.join(home_base, proxy_base)
    out_dir = options.dataDirLocal + "/" + options.tag + "/{}/"
    out_dir_xrd = "/" + username + "/SUEP/" + options.tag + "/{}/"
    move_dir = "/work/submit/" + username + "/SUEP/" + options.tag + "/{}/"
    jobs_base_dir = "/work/submit/" + username + "/SUEP/logs/"

    if options.move:
        if not os.path.isdir("/work/submit/" + username + "/SUEP/" + options.tag):
            subprocess.run(
                [
                    "mkdir",
                    "/work/submit/" + username + "/SUEP/" + options.tag,
                ]
            )

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
        logging.info(f"--- proxy lifetime is {lifetime} hours")
        if lifetime < 10.0:  # we want at least 100 hours
            logging.warning("--- proxy has expired !")
            regenerate_proxy = True

    if regenerate_proxy:
        redone_proxy = False
        while not redone_proxy:
            status = os.system("voms-proxy-init -voms cms")
            if os.WEXITSTATUS(status) == 0:
                redone_proxy = True
        copyfile("/tmp/" + proxy_base, proxy_copy)

    with open(options.input) as stream:
        totals, completeds = 0, 0
        missing_samples = []  # samples with no inputfiles.dat or output dir
        empty_samples = []  # samples with  no completed jobs
        for isample, sample in enumerate(stream.read().split("\n")):
            if len(sample) <= 1:
                continue
            if "#" in sample:
                continue
            if "/" in sample and len(sample.split("/")) <= 1:
                continue

            sample_name = sample
            if "/" in sample_name:
                sample_name = sample_name.split("/")[-1]
            if ".root" in sample_name:
                sample_name = sample_name.replace(".root", "")

            data[sample_name] = {}

            jobs_dir = "/".join([options.tag, sample_name])
            jobs_dir = jobs_base_dir + jobs_dir

            if not os.path.isdir(out_dir.format(sample_name)):
                logging.warning("Cannot find " + out_dir.format(sample_name))
                missing_samples.append(sample_name)
                continue

            # delete files that are corrupted (i.e., empty)
            for file in os.listdir(out_dir.format(sample_name)):
                size = os.path.getsize(out_dir.format(sample_name) + "/" + file)
                if size == 0:
                    subprocess.run(["rm", out_dir.format(sample_name) + "/" + file])

            logging.info(jobs_dir)

            # We write the original list. inputfiles.dat will now contain missing files. Compare with original list
            if not os.path.isfile(jobs_dir + "/" + "inputfiles.dat"):
                logging.warning("Cannot find " + jobs_dir + "/" + "inputfiles.dat")
                missing_samples.append(sample_name)
                continue
            if os.path.isfile(jobs_dir + "/" + "original_inputfiles.dat") != True:
                copyfile(
                    jobs_dir + "/" + "inputfiles.dat",
                    jobs_dir + "/" + "original_inputfiles.dat",
                )

            # Find out the jobs that run vs the ones that failed
            jobs = [
                line.rstrip()
                for line in open(jobs_dir + "/" + "original_inputfiles.dat")
            ]

            njobs = len(jobs)
            complete_list = os.listdir(out_dir.format(sample_name))
            complete_list = [f for f in complete_list if not f.startswith("gitinfo")]
            nfile = len(complete_list)

            data[sample_name]['total'] = njobs
            data[sample_name]['completed'] = nfile

            if njobs == 0:
                missing_samples.append(sample)
                continue

            if nfile == 0:
                empty_samples.append(sample)

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
            completeds += nfile
            totals += njobs

            # If files are missing we resubmit with the same condor.sub
            if options.resubmit and (nfile < njobs) and not (options.auto):
                logging.info(f"-- resubmitting files for {sample}")
                file_names = []
                for item in complete_list:
                    if "." not in item:
                        continue
                    file_names.append(os.path.splitext(item)[0])

                jobs_resubmit = [
                    item for item in jobs if item.split("\t")[-1] not in file_names
                ]
                resubmit_file = open(jobs_dir + "/" + "inputfiles.dat", "w")
                for redo_file in jobs_resubmit:
                    resubmit_file.write(redo_file + "\n")
                resubmit_file.close()

                if options.wait > 0 and isample != 0:
                    logging.info(
                        f"Waiting {options.wait} hours ({round(options.wait*60)} minutes) before resubmitting"
                    )
                    subprocess.run(["sleep", str(options.wait * 3600)])

                htc = subprocess.Popen(
                    "condor_submit " + os.path.join(jobs_dir, "condor.sub"),
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    close_fds=True,
                )
                out, err = htc.communicate()
                exit_status = htc.returncode
                logging.info(f"condor submission status : {exit_status}")

            if options.move:
                if not os.path.isdir(move_dir.format(sample_name)):
                    os.system("mkdir " + move_dir.format(sample_name))

                # delete files that are corrupted (i.e., empty)
                for file in os.listdir(move_dir.format(sample_name)):
                    size = os.path.getsize(move_dir.format(sample_name) + "/" + file)
                    if size == 0:
                        subprocess.run(
                            ["rm", move_dir.format(sample_name) + "/" + file]
                        )

                # get list of files already in /work
                movedFiles = os.listdir(move_dir.format(sample_name))

                # get list of files in T3
                allFiles = os.listdir(out_dir.format(sample_name))

                # get list of files missing from /work that are in T3
                filesToMove = list(set(allFiles) - set(movedFiles))

                # move those files
                logging.info(
                    "Moving "
                    + str(len(filesToMove))
                    + " files to "
                    + move_dir.format(sample_name)
                )
                for file in filesToMove:
                    subprocess.run(
                        [
                            "xrdcp",
                            options.redirector + "/" + out_dir_xrd + file,
                            move_dir.format(sample_name) + "/",
                        ]
                    )

        logging.info("")
        logging.info("")
        logging.info("TOTAL")
        percent = completeds / totals * 100
        logging.info(
            colored("\t\t --> completed", "green")
            if completeds == totals
            else colored(
                "\t --> ({}/{}) finished. {:.1f}% complete".format(
                    completeds, totals, percent
                ),
                "red",
            )
        )

        if len(missing_samples) > 0:
            logging.info("")
            logging.info("The following samples were missing:")
            for s in missing_samples:
                logging.info(s)

        if len(empty_samples) > 0:
            logging.info("")
            logging.info("The following samples had no completed jobs:")
            for s in empty_samples:
                logging.info(s)

    return data


if __name__ == "__main__":
    main()
