#! /bin/bash
export MOVE_ON=true
#### Root folder from dropbox 
export ROOT_PATH="/home/luis/Dropbox/DATASETS_AUDIO/Unsupervised_Pipeline"
export SRC_PATH=$(pwd)

export EXP_NAME="EXP010"
# export DATASET_NAME="TTS4_easy"
export DATASET_NAME="TestAO-Irma"


export VAD_NAME="SHAS"
export FEAT_NAME="DV"
export METHOD_NAME="hdb"
export AZURE_FLAG=false

## Optional methods
export DOUBLE_TALK_FLAG=true
export SILENT_DET_FLAG=true
export ENHANCE_FLAG=false

## If GT_CSV_FOLDER is provided, predict_only will be set to false
export GT_CSV_FOLDER="${ROOT_PATH}/${DATASET_NAME}/GT_final/"
if [ -d "$GT_CSV_FOLDER" ]; then
    export PREDICT_ONLY=false
else
    export PREDICT_ONLY=true
fi
echo -e "predict_only: $PREDICT_ONLY"

export USE_PKL_LABEL=true

## External Repositories used
export VAD_LOCATION="/home/luis/Dropbox/SpeechSpring2023/shas"

## Pretrained models used:
export STG1_DT_PRETRAINED="${SRC_PATH}/pre-trained/best_overlap_detection_model_xvectors_May20.pth"
export STG1_VAD_PRETRAINED="${VAD_LOCATION}/en_sfc_model_epoch-6.pt"
# export STG2_ENH_PRETRAINED="${SRC_PATH}/pre-trained/model_S1_256_med_betaNO3_mask00_lr-5_ep180_73.pth"
export STG2_ENH_PRETRAINED="${SRC_PATH}/pre-trained/model_S1_256_hard_betaNO4_mask00_lr-5_ep180_73.pth"
# export STG2_ENH_PRETRAINED="${SRC_PATH}/pre-trained/model_S1_easyN1_mask00_lr-5_ep180_73.pth"

## Segmentation Parameters
export seg_ln="1.0"
export step_size="0.3"
export gap_size="0.4"
export consc_th="1"
export min_overlap_percentage="0.3"

export DT_THRESHOLD="0.8"

export STG1_MP4_FOLDER="${ROOT_PATH}/${DATASET_NAME}/input_mp4s"

#### Stage 1 VAD
export current_stg1="${ROOT_PATH}/${DATASET_NAME}/STG_1/STG1_${VAD_NAME}"
export STG1_WAVS="${ROOT_PATH}/${DATASET_NAME}/input_wavs/"
export STG1_FILTERED_CHUNKS_WAVS="${current_stg1}/wav_chunks_filtered"

# source ./BB_Stages_bash/STG1_SHAS.sh

#### Stage 2 Feature Extraction
export current_stg2="${ROOT_PATH}/${DATASET_NAME}/STG_2/STG2_${EXP_NAME}-${VAD_NAME}-${FEAT_NAME}"

export STG2_FEATS_PICKLE="${current_stg2}/${DATASET_NAME}_${VAD_NAME}_${FEAT_NAME}_feats.pickle"
export STG2_FEATS_ENHANCED="${current_stg2}/${DATASET_NAME}_${VAD_NAME}_${FEAT_NAME}_featsEN.pickle"

export ENHANCE_RUN_ID="skipped"

# if [ "$MOVE_ON" = "true" ]; then
# source ./BB_Stages_bash/STG2_DVECTORS_ENHANCER.sh
# fi

#### Stage 3 Unsupervised Method
export current_stg3="${ROOT_PATH}/${DATASET_NAME}/STG_3/STG3_${EXP_NAME}-${VAD_NAME}-${FEAT_NAME}-${METHOD_NAME}"
export STG3_MERGED_WAVS="${current_stg3}/merged_wavs"
export STG3_AL_INPUT="${current_stg3}/AL_input_merged.csv"
export STG3_FINAL_CSV="${current_stg3}/pred_output_${EXP_NAME}"

export pca_elem="0"

export min_cluster_size="25"
export hdb_mode="eom"
export min_samples="5"

export RUN_PARAMS="pca${pca_elem}_mcs${min_cluster_size}_ms${min_samples}_${hdb_mode}"

# cd $SRC_PATH
# if [ "$MOVE_ON" = "true" ]; then
# source ./BB_Stages_bash/STG3_META_HDB.sh
# fi


#### Stage 4 LP
export LP_METHOD_NAME="LP1"
export current_stg4="${ROOT_PATH}/${DATASET_NAME}/STG_4/STG4_${LP_METHOD_NAME}"
export STG4_HUMAN="${current_stg4}/webapp_results"


cd $SRC_PATH
if [ "$MOVE_ON" = "true" ]; then
    source ./BB_Stages_bash/STG4_LP.sh
fi

## Add Azure comparison

#### GetBack to where you once belonged

# cp $SRC_PATH/EXP001_HDBSCAN_script_luis.sh $current_stg3/script_contents.txt
conda activate
cd $SRC_PATH

# # Unset all the variables
# unset ROOT_PATH
# unset SRC_PATH
# unset EXP_NAME
# unset DATASET_NAME
# unset SHAS_NAME
# unset FEAT_NAME
# unset METHOD_NAME
# unset STG1_WAVS
# unset STG1_FINAL_CSV
# unset current_stg2
# unset STG2_FEATS_PICKLE
# unset current_stg3
# unset STG3_MERGED_WAVS
# unset min_cluster_size
# unset pca_elem
# unset hdb_mode
# unset min_samples
# unset RUN_ID
# unset RUN_PARAMS

# unset SHAS_LOCATION
# unset path_to_yaml_folder
# unset path_to_yaml_file
# unset SHAS_ROOT
# unset path_to_checkpoint
# unset STG1_WAVS
# unset STG1_FINAL_CSV
# unset SRC_PATH

# unset HDBSCAN_LOCATION
# unset STG2_CHUNKS_WAVS
# unset STG2_MFCC_FILES
# unset SRC_PATH
# unset STG2_FEATS_PICKLE

# unset HDBSCAN_LOCATION
# unset STG3_HDBSCAN_PRED_OUTPUT
# unset STG3_SEPARATED_WAVS
# unset STG3_OUTLIERS_WAVS
# unset STG4_FINAL_CSV
# unset STG4_SEPARATED_MERGED_WAVS