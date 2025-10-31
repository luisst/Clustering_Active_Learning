#! /bin/bash
export SKIP_VAD=false
export SKIP_1A=false
export SKIP_1C=false
export SKIP_1D=false
export SKIP_1E=false

# Add SRC_PATH to PYTHONPATH for Python module imports
export PYTHONPATH="${SRC_PATH}:${PYTHONPATH}"

### Stage 1: VAD Run SHAS on the test set

echo "1) Running VAD on the test set"

cd $VAD_LOCATION

# Change Conda environment
conda activate shas

export STG1_VAD_CSV="${current_stg1}/shas_output_csv/"
export path_to_yaml_folder="${current_stg1}/shas_output_yml"
export STG1_RAW_CHUNKS_WAVS="${current_stg1}/wav_chunks_raw"
export STG1_CHUNKS_WAVS="${current_stg1}/wav_chunks"

python3 ${SRC_PATH}/folder_verify.py $path_to_yaml_folder
if [ $? -eq 1 ]; then
    export SKIP_VAD=true
fi
python3 ${SRC_PATH}/folder_verify.py $STG1_VAD_CSV
if [ $? -eq 1 ]; then
    export SKIP_1A=true
fi
python3 ${SRC_PATH}/folder_verify.py $STG1_RAW_CHUNKS_WAVS
if [ $? -eq 1 ]; then
    export SKIP_1C=true
fi
python3 ${SRC_PATH}/folder_verify.py $STG1_CHUNKS_WAVS
if [ $? -eq 1 ]; then
    export SKIP_1D=true
fi
python3 ${SRC_PATH}/folder_verify.py $STG1_FILTERED_CHUNKS_WAVS
if [ $? -eq 1 ]; then
    export SKIP_1E=true
fi

export VAD_ROOT="${VAD_LOCATION}/repo"
export path_to_yaml_file="${path_to_yaml_folder}/VAD_${EXP_NAME}.yaml"


# echo -e "\n\t>>>>> Stage1vad: $STG1_WAVS\n"
# if [ "$SKIP_VAD" != "true" ]; then
#     python3 ${VAD_ROOT}/src/supervised_hybrid/segment.py -wavs $STG1_WAVS -ckpt $STG1_VAD_PRETRAINED -yaml $path_to_yaml_file -max 10
#     echo -e "\t>>>>> VAD output: $path_to_yaml_file"
# else
#     echo -e "\n\t>>>>> Stage1: VAD skipped\n"
# fi

# echo -e "\n\t>>>>> Stage1a: $path_to_yaml_file\n"
# cd $SRC_PATH
# if [ "$SKIP_1A" != "true" ]; then
#     python3 ${SRC_PATH}/01_VAD_chunks/Stage1a_convert_shasYML_csv.py $path_to_yaml_file $STG1_VAD_CSV
#     echo -e "\t>>>>> Converted to CSV: $STG1_VAD_CSV"
# else
#     echo -e "\n\t>>>>> Stage1a: Conversion skipped\n"
# fi

# if [ $? -ne 0 ]; then
#     export MOVE_ON=false
#     echo "Move on: $MOVE_ON"
#     return 1
# fi

# echo -e "before Metrics"

# if [ "$PREDICT_ONLY" = "true" ] && [ "$MOVE_ON" = "true" ]; then
# echo -e "\n\t>>>>> Stage 1b: Metrics \n"
# export VAD_metric_folder="${ROOT_PATH}/${DATASET_NAME}/STG_1/STG1_${VAD_NAME}/metrics"
# python3 ${SRC_PATH}/folder_verify.py $VAD_metric_folder

# export VAD_pred_ext="txt"
# export vad_method="shas"

# python3 ${SRC_PATH}/01_VAD_chunks/Stage1b_metric_vad.py --csv_pred_folder $STG1_VAD_CSV\
#  --GT_csv_folder $GT_CSV_FOLDER\
#  --audios_folder $STG1_WAVS\
#  --metric_output_folder $VAD_metric_folder\
#  --pred_extensions $VAD_pred_ext\
#  --method_name $vad_method\
#  --run_name $VAD_NAME
# fi


echo -e "\n\t>>>>> Stage1c: Divide into chunks $STG1_WAVS \n"

echo "Azure flag is: $AZURE_FLAG"
echo "Minimum overlap percentage: $min_overlap_percentage"

if [ "$SKIP_1C" != "true" ] && [ "$MOVE_ON" = "true" ]; then
    python3 ${SRC_PATH}/01_VAD_chunks/Stage1c_divide_into_chunks.py --stg1_wavs $STG1_WAVS\
     --stg1_final_csv $STG1_VAD_CSV \
     --stg1_chunks_wavs $STG1_RAW_CHUNKS_WAVS\
     --ln $seg_ln --st $step_size --min_overlap_pert $min_overlap_percentage\
     --GT_folder_path $GT_CSV_FOLDER
else
    echo -e "\n\t>>>>> Stage1c: Chunk division skipped\n"
fi
# Check if the Python script was successful
if [ $? -ne 0 ]; then
    export MOVE_ON=false
    echo "Move on: $MOVE_ON"
    return 1
fi

echo -e "\n\t>>>>> Stage 1d: Silent Detector $STG1_RAW_CHUNKS_WAVS\n"
conda activate metaSR3
export keep_perc="90"
if [ "$SKIP_1D" != "true" ] && [ "$MOVE_ON" = "true" ]; then

    if [ "$DOUBLE_TALK_FLAG" = "true" ]; then
        python3 ${SRC_PATH}/01_VAD_chunks/Stage1d_silent_detector.py --input_chunks_folder $STG1_RAW_CHUNKS_WAVS\
        --filtered_wavs_folder $STG1_CHUNKS_WAVS\
        --keep_perc $keep_perc

        # Check if the Python script was successful
        if [ $? -ne 0 ]; then
            export MOVE_ON=false
            echo "Move on: $MOVE_ON"
            return 1
        fi
    else
        echo -e "\n\t>>>>> Silent detection SKIPPED! \n"
        cp -r $STG1_RAW_CHUNKS_WAVS/* $STG1_CHUNKS_WAVS
        # Check if the copy was successful
        if [ $? -ne 0 ]; then
            export MOVE_ON=false
            echo "Move on: $MOVE_ON"
            return 1
        fi
    fi
else
    echo -e "\n\t>>>>> Stage 1d: Silent detection skipped\n"
fi



echo -e "\t>>>>> Stage 1e: Double-Talk detection $STG1_CHUNKS_WAVS"
if [ "$SKIP_1E" != "true" ]; then

    if [ "$DOUBLE_TALK_FLAG" = "true" ]; then
        conda activate dtp11
        python3 ${SRC_PATH}/01_VAD_chunks/Stage1e_DT_detection.py --stg1_chunks_wavs $STG1_CHUNKS_WAVS\
        --stg1_dt_pretrained $STG1_DT_PRETRAINED \
        --stg1_filtered_chunks_wavs $STG1_FILTERED_CHUNKS_WAVS \
        --stg1_dt_th $DT_THRESHOLD

        # Check if the Python script was successful
        if [ $? -ne 0 ]; then
            export MOVE_ON=false
            echo "Move on: $MOVE_ON"
            return 1
        fi
    else
        echo -e "\n\t>>>>> Double-Talk detection SKIPPED! \n"
        cp -r $STG1_CHUNKS_WAVS/* $STG1_FILTERED_CHUNKS_WAVS
        # Check if the copy was successful
        if [ $? -ne 0 ]; then
            export MOVE_ON=false
            echo "Move on: $MOVE_ON"
            return 1
        fi
    fi
else
    echo -e "\n\t>>>>> Stage 1e: DT detection skipped\n"
fi
