"""
Run splitTrees.py on unsplit SUEP samples on T2

Author: Pietro Lugato
"""

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="signal splitting inputs")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data.txt",
        help="input datasets",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="/data/submit/pmlugato/SUEP/split_signals_temp/",
        help="output directory",
        required=False,
    )
    options = parser.parse_args()

    with open(options.input) as stream:

        for sample_path in stream.read().split("\n"):

            root_files = []
            cmd = ["xrdfs", "root://xrootd5.cmsaf.mit.edu/", "ls", sample_path]
            comm = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
            )

            return_code = comm.wait()

            print(f"xrdfs root://xrootd5.cmsaf.mit.edu/ ls {sample_path}")

            root_files = comm.communicate()[0].decode("utf-8").split("\n")
            full_path_root_files = []
            for f in root_files:
                if len(f) == 0:
                    continue
                full_f = f"root://xrootd5.cmsaf.mit.edu/{f}"
                full_path_root_files.append(full_f)

            with open("temp_list.txt", "w") as outfile:
                outfile.write("\n".join(str(i) for i in full_path_root_files))

            splitTrees_run = subprocess.run(
                [
                    "python",
                    "splitTrees.py",
                    "--inputFiles",
                    "temp_list.txt",
                    "--output",
                    options.output,
                    "--jobs",
                    "10",
                ]
            )


if __name__ == "__main__":
    main()
