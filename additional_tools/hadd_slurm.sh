#!/bin/bash
# Use to combine hadd every .root file in every subfolder of a parent directory that you specify
# Usage: ./hadd_subfolders.sh <parent_directory>

parent_dir="$1"
outdir="$2"
if [ -z "$parent_dir" ]; then
    echo "Usage: $0 <parent_directory>"
    exit 1
fi
if [ ! -d "$parent_dir" ]; then
    echo "Error: $parent_dir is not a directory."
    exit 1
fi
if [ -z "$outdir" ]; then
    echo "Usage: $0 <parent_directory> <output_directory>"
    exit 1
fi
if [ ! -d "$outdir" ]; then
    echo "Error: $outdir is not a directory."
    exit 1
fi

finished_count=0
bad_count=0
tot_count=0
submitted_count=0
for subdir in "$parent_dir"/*/; do

    ((tot_count++))

    subdir="$subdir/NANOAOD/"

    echo "Processing directory: $subdir"
    sample=$(basename "$(dirname "$subdir")")
    merged_file="$outdir/$sample/$sample.root"

    if [ ! -d "$outdir/$sample" ]; then
        mkdir -p "$outdir/$sample"
    fi

    if [ -f "$merged_file" ]; then
        echo "Exists: $merged_file"
        file_size=$(stat -c%s "$merged_file")
        if [ "$file_size" -gt $((1 * 1024 * 1024 * 1024)) ]; then
            echo "File size of $merged_file is larger than 1 GB."
            echo "Probably good, will not re-run."
            echo
            ((finished_count++))
            continue
        else
            echo "File size of $merged_file is less than or equal to 1 GB."
            echo "Likely fucked up, will re-run."
            echo
            rm "$merged_file"
            ((bad_count++))
        fi

    fi

    echo "Submitting job"

    # Make a unique job script
    job_script=$(mktemp)
    cat <<EOF > "$job_script"
#!/bin/bash
#SBATCH --job-name=hadd_job
#SBATCH --output=/work/submit/lavezzo/SUEP/logs/hadd/hadd_%j.log
#SBATCH --error=/work/submit/lavezzo/SUEP/logs/hadd/hadd_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --partition=submit,submit-1gbs
#SBATCH --cpus-per-task=4
#SBATCH --exclude=submit01,submit07 # don't know why I get permission denieds on submit01

echo "made it to host $(hostname)"
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /home/submit/lavezzo/SUEP/CMSSW_14_1_0_pre4/src/SUEPLimits # just need some CMS release
cmsenv
cd ~
hadd -f -k -j 4 "$merged_file" "$subdir"/nano*.root

echo "All done!"
EOF

    sbatch "$job_script"
    echo "Submitted"
    echo

    ((submitted_count++))

    #echo "Job script:"
    #echo "$job_script"
    #break

done


echo "Finished count: $finished_count"
echo "Bad count: $bad_count"
echo "Total count: $tot_count"
echo "Submitted count: $submitted_count"
