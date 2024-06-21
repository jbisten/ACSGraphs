##!/bin/bash
# Initialize variable
BIDS_DIR=""
num_processes=20
SUBJECT=""  # New variable for the optional subject argument

# Function to wait for all background jobs to finish
wait_for_jobs() {
    while [[ $(jobs -r -p | wc -l) -gt 0 ]]; do
        wait -n
    done
}

# Loop through arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift ;;  # If --data is found, set BIDS_DIR to the next argument
        --subject) SUBJECT="$2"; shift ;;  # New case for --subject
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift  # Move to next argument
done

# Check if required variable was set
if [[ -z "$CONFIG" ]]; then
    echo "Usage: $0 --config CONFIG [--subject SUBJECT_ID]"
    exit 1
fi

########################################################
#
#                    Config Processing
#
########################################################
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "MNI Clustering Script Hemispherotomy Patients"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Processing config: $CONFIG"
# Extract the database host
bids_dir=$(yq '.bids_dir' $CONFIG | sed 's/"//g')
participants_tsv=$(yq '.participants_tsv' $CONFIG | sed 's/"//g')
seeding=$(yq '.seeding' $CONFIG | sed 's/"//g')

echo "Seeding Method: ${seeding}"
sleep 2

# Define subject directory in the BIDS dataset
SOURCEDATA="$bids_dir/sourcedata"
DERIVATIVES="$bids_dir/derivatives"

mkdir -p "$DERIVATIVES"

declare -a ukb_ids
declare -a bids_ids

# TODO: Replace participants.tsv extraction with something smart considering the yaml file
mapfile -t bids_ids < <(python ./utils/extract_participants.py $participants_tsv)

# If SUBJECT is set, only process that subject
if [[ -n "$SUBJECT" ]]; then
    bids_ids=("$SUBJECT")
fi

# Create Derivatives Working Directory
cluster_dir="${DERIVATIVES}/clustering"
mkdir -p ${cluster_dir}
    
# Run ICP
CMD="python ./utils/icp.py"

# Loop over files matching the pattern in the directory
for file in ${cluster_dir}/*_mni_final_af_*.tck; do
    if [ -f "$file" ]; then  # Check if it is a file and not an empty result
        # Form the corresponding _icp_ version of the file name
        icp_file="${file/_mni_final_af_/_mni_icp_final_af_}"
        if [ ! -f "$icp_file" ]; then  # Check if the corresponding _icp_ file does not exist
            CMD+=" --infiles $file"
        fi
    fi
done

CMD+=" -f"

# Execute the command
# echo "Executing: $CMD"
eval $CMD

# Create algorithm dir
mkdir  -p ${cluster_dir}/hdbscan_results/
mkdir  -p ${cluster_dir}/kmeans_results/
mkdir  -p ${cluster_dir}/quickbundle_results/


# Run ICP
CMD="python ./kmeans/main.py"

# Loop over files matching the pattern in the directory
for file in ${cluster_dir}/*_mni_icp_final_af_*.tck; do
    if [ -f "$file" ]; then  # Check if it is a file and not an empty result
            CMD+=" --infiles $file"
    fi
done

for file in ${cluster_dir}/*.h5; do
    if [ -f "$file" ]; then  # Check if it is a file and not an empty result
            CMD+=" --inhdf5s $file"
    fi
done

CMD+=" --k 3 --outdir ${cluster_dir}/kmeans_results/"

eval $CMD


#python ./kmeans/main.py --infiles ${cluster_dir}/mni_af_icp.tck --hdf5in --outdir $cluster_dir/kmeans_results/
#python ./hdbscan/main.py --infile ${cluster_dir}/mni_af_icp.ply --outdir $cluster_dir/hdbscan_results/ 

echo "Done"

