#!/bin/bash
# Initialize variable
BIDS_DIR=""
num_processes=20
SUBJECT=""  # New variable for the optional subject argument


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
#					Config Processing
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
kmeans="${DERIVATIVES}/kmeans"
mkdir -p ${kmeans}
for sub in "${bids_ids[@]}"; do
    subject_dir="$bids_dir/$sub"

    for session_dir in "$subject_dir"/*/; do
	session=$(basename "$session_dir")
    session_dwi="$subject_dir/$session/dwi"
    session_anat="$subject_dir/$session/anat"
    preproc="$DERIVATIVES/$sub/$session/dwi/preproc"
    tracts="$DERIVATIVES/$sub/$session/dwi/tracts"
    mni="$DERIVATIVES/$sub/$session/dwi/mni"
    t1_brain_template="/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
    dti_FA="${preproc}/dti_FA.nii.gz"
    t1=$(find "$session_anat/" -maxdepth 1 -name "*T1w*.nii.gz" -type f)

	####################################################
	# 		Extract the variables for each subject
	#################################################### 
    IFS=$'\t' read -r group fmri id affected_hemisphere excluded<<< $(python ./utils/get_subject_meta.py ${participants_tsv} ${sub})

    if [[ "$excluded" == "yes" ]]; then
		echo "Skipping subject ${sub}, excluded . . ."
		continue
	fi

    if [[ "$fmri" -eq 0 ]] && [[ "$seeding" == "fmri" ]]; then
		echo "Skipping subject ${sub}, no fmri data . . ."
		continue
	fi

    echo "Processing Subject: $sub"
    sides=('lh' 'rh')

    # Warp Target Left AF to MNI
    for h in "${sides[@]}"; do 
        if [[ "$affected_hemisphere" == ${h} || $h == "nan" ]]; then
		    echo "Skipping ${sub} hemisphere (${h}): hemisphere is affected . . ."
		    continue
	    fi
        (

        ########################################
        #         Warping streamlines
        ########################################
        tck_in="${tracts}/ukft_af_${h}.tck"
        tck_out="${tracts}/mni_ukft_af_${h}.tck"

        mean_B0_to_t1="${mni}/t1_to_mean_B0_0GenericAffine.mat"
        t1_to_t1_mni_affine="${mni}/t1_to_t1_mni_0GenericAffine.mat"
        t1_to_t1_mni_warp="${mni}/t1_to_t1_mni_1Warp.nii.gz"

        # TODO: Adapt warpstreams so that it can write .tck and .ply
        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        python3 ./utils/warpstreams.py\
            -i ${tck_in}\
            -r ${t1_brain_template}\
            -t "${mean_B0_to_t1}@0"\
            -t "${t1_to_t1_mni_affine}@1"\
            -t "${t1_to_t1_mni_warp}@1"\
            -o ${tck_out}

        # TODO: we can then ommit this step
        # Convert to .ply and move to ffclust dir
        tractconv -i ${tck_out} -o ${kmeans}/${sub}_mni_ukft_af_${h}.ply -r ${t1_brain_template} 

        # Calculate radial diffusivity
		fslmaths ${preproc}/dti_L2.nii.gz -add ${preproc}/dti_L3.nii.gz -div 2 ${preproc}/dti_RD.nii.gz 
        fslmaths ${preproc}/dti_L1.nii.gz -add ${preproc}/dti_L2.nii.gz -add ${preproc}/dti_L3.nii.gz ${preproc}/dti_TR.nii.gz 
        
        # TODO: Enrich .ply with features
        python ./utils/augmentply.py --inply "${kmeans}/${sub}_mni_ukft_af_${h}.ply" \
                --outply "${kmeans}/${sub}_mni_ukft_af_${h}.ply" \
                --fa "${preproc}/dti_FA.nii.gz" \
                --ad "${preproc}/dti_L1.nii.gz" \
                --rd "${preproc}/dti_RD.nii.gz" \
                --md "${preproc}/dti_MD.nii.gz" \
                --tr "${preproc}/dti_TR.nii.gz" \
                --l1 "${preproc}/dti_L1.nii.gz" \
                --l2 "${preproc}/dti_L2.nii.gz" \
                --l3 "${preproc}/dti_L3.nii.gz" 

        ) &

        if [[ $(jobs -r -p | wc -l) -ge $num_processes ]]; then
            wait -n
        fi
        done
    done
done

for h in "${sides[@]}"; do (

    CMD="python ./utils/tcks2ply.py"

    # Loop over files matching the pattern in the directory
    for file in ${kmeans}/*_mni_ukft_af_${h}.tck; do
        if [ -f "$file" ]; then  # Check if it is a file and not an empty result
            CMD+=" --infiles $file"
        fi
    done

    CMD+=" --outfile $kmeans/mni_af_${h}.ply -f"

    # Execute the command
    echo "Executing: $CMD"
    eval $CMD

    # Cluster
    python ./kmeans/main.py --infile ${kmeans}/mni_af_${h}.ply --outdir $kmeans/results_${h}/ 

    ) &

    if [[ $(jobs -r -p | wc -l) -ge $num_processes ]]; then
        wait -n
    fi
 
done



