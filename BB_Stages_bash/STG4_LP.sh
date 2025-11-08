#! /bin/bash

export MOVE_ON=true

# Verify/create Stage 4 folders
python3 ${SRC_PATH}/folder_verify.py $STG4_HUMAN

# python3 ${SRC_PATH}/folder_verify.py $STG3_FINAL_CSV
# if [ $? -eq 1 ]; then
#     export SKIP_STG3e=true
# fi

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

# ## Create the final csv file
# echo -e "\n\t>>>>> Stage 3e: Output Final CSV prediction $STG3_FINAL_CSV\n"
# if [ "$SKIP_STG3e" != "true" ]; then
#     python3 ${SRC_PATH}/03_Clustering_TDA/Stg3e_create_csv_from_merged.py --stg3_merged_wavs $STG3_MERGED_WAVS\
#     --stg3_final_csv $STG3_FINAL_CSV --stg3_separated_merged_wavs $STG3_SEPARATED_MERGED_WAVS

#     # Check if the Python script was successful
#     if [ $? -ne 0 ]; then
#         export MOVE_ON=false
#         echo "Move on: $MOVE_ON"
#         return 1
#     fi
# else
#     echo -e "\n\t>>>>> Stage 3e: Final CSV prediction skipped\n"
# fi

