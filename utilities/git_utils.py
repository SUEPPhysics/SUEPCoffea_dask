import os
import subprocess


def get_git_info(path="."):
    """
    Get the current commit and git diff.
    """

    # Change directory to the git repo
    os.chdir(path)

    # Get the current commit and diff
    commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    diff = (
        subprocess.check_output(["git", "diff", "--", ".", "':(exclude)*.ipynb'"])
        .strip()
        .decode("utf-8")
    )

    return commit, diff


def write_git_info(path="."):
    """
    Write the current commit and git diff to a file.
    """
    import datetime

    commit, diff = get_git_info()
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    ofile = os.path.join(path, f"gitinfo_{formatted_datetime}.txt")
    with open(ofile, "w") as gitinfo:
        gitinfo.write("Commit: \n" + commit + "\n")
        gitinfo.write("Diff: \n" + diff + "\n")
        gitinfo.close()
    return ofile
