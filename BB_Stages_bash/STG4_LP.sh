#! /bin/bash

export MOVE_ON=true

## Validate speaker JSON consistency across dataset folders
echo -e "\n\t>>>>> Validating Speaker JSON Consistency\n"

# Build validation command with optional folders
VALIDATION_CMD="python3 ${SRC_PATH}/04_Active_learning_loop/Stg4_validate_speaker_json.py --json_filename ${SPEAKERS_JSON_FILENAME}"

# Add GT folder if it exists
if [ -d "${GT_CSV_FOLDER}" ]; then
    VALIDATION_CMD="${VALIDATION_CMD} --gt_folder ${GT_CSV_FOLDER}"
fi

# Add input_wavs folder if it exists
if [ -d "${STG1_WAVS}" ]; then
    VALIDATION_CMD="${VALIDATION_CMD} --input_wavs ${STG1_WAVS}"
fi

# Add input_mp4s folder if it exists
if [ -d "${STG1_MP4_FOLDER}" ]; then
    VALIDATION_CMD="${VALIDATION_CMD} --input_mp4s ${STG1_MP4_FOLDER}"
fi

# Execute validation
eval ${VALIDATION_CMD}

# Check if validation was successful
if [ $? -ne 0 ]; then
    echo -e "\n\t ✗ ERROR: Speaker JSON validation failed"
    echo -e "\tPlease ensure ${SPEAKERS_JSON_FILENAME} files are consistent across folders"
    echo -e "\tYou may need to regenerate them using the appropriate script\n"
    export MOVE_ON=false
    return 1
fi

# Verify/create Stage 4 folders
python3 ${SRC_PATH}/folder_create.py $STG4_HUMAN

## Generate Active Learning input files from merged samples
echo -e "\n\t>>>>> Stage 4a: Generate AL Input Files from Merged Samples\n"
echo -e "  - AL CSV input: $STG3_AL_INPUT"
echo -e "  - Merged HDF5: $STG3_MERGED_H5"
echo -e "  - Output folder: $current_stg4\n"

python3 ${SRC_PATH}/04_Active_learning_loop/stg4a_generate_ALinputs.py \
    --stg3_al_input $STG3_AL_INPUT \
    --merged_dataset_h5 $STG3_MERGED_H5 \
    --stg4_al_folder $current_stg4

# Check if the Python script was successful
if [ $? -ne 0 ]; then
    export MOVE_ON=false
    echo "Move on: $MOVE_ON"
    return 1
fi


# echo -e "\n\t>>>>> Stg4b Marker webapp: $STG4_AL_INPUT\n"
# python3 ${SRC_PATH}/04_Active_learning_loop/stg4b_marker_webapp.py --stg1_mp4_candidate $STG1_MP4_FOLDER\
#     --stg4_al_folder $current_stg4


# Unset all stage 4 variables (MOVE_ON is set here but unset in main script)
# No additional variables unique to stage 4 besides those in main script
