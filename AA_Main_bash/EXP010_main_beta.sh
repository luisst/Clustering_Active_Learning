#! /bin/bash
export MOVE_ON=true
#### Root folder from dropbox 
export ROOT_PATH="/home/luis/Dropbox/DATASETS_AUDIO/Unsupervised_Pipeline"
export SRC_PATH=$(pwd)

export EXP_NAME="EXP010"
export DATASET_NAME="TestAO-Irma"

export VAD_NAME="SHAS"
export FEAT_NAME="DV"
export METHOD_NAME="uH10NoEn"

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

## External Repositories used
export VAD_LOCATION="/home/luis/Dropbox/SpeechSpring2023/shas"

## Pretrained models used:
export STG1_DT_PRETRAINED="${SRC_PATH}/pre-trained/best_overlap_detection_model_xvectors_May20.pth"
export STG1_VAD_PRETRAINED="${VAD_LOCATION}/en_sfc_model_epoch-6.pth"
export STG2_ENH_PRETRAINED="${SRC_PATH}/pre-trained/model_mask00_ep180_betaAug18_73.pth"

## Segmentation Parameters
export seg_ln="1.0"
export step_size="0.3"
export gap_size="0.4"
export consc_th="1"

export DT_THRESHOLD="0.8"

echo -e "\t>>>>> STG-1 VAD, Chunks Divide, Silent Detection, Double-Talk Detection <<<<<"

#### Stage 1 VAD
export current_stg1="${ROOT_PATH}/${DATASET_NAME}/STG_1/STG1_${VAD_NAME}"
export STG1_WAVS="${ROOT_PATH}/${DATASET_NAME}/input_wavs/"
export STG1_FILTERED_CHUNKS_WAVS="${current_stg1}/wav_chunks_filtered"

# source ./BB_Stages_bash/STG1_SHAS.sh

#### Stage 2 Feature Extraction
export current_stg2="${ROOT_PATH}/${DATASET_NAME}/STG_2/STG2_${EXP_NAME}-${VAD_NAME}-${FEAT_NAME}"

export STG2_FEATS_PICKLE="${current_stg2}/${DATASET_NAME}_${VAD_NAME}_${FEAT_NAME}_feats.pickle"
export STG2_FEATS_ENHANCED="${current_stg2}/${DATASET_NAME}_${VAD_NAME}_${FEAT_NAME}_featsEN-false.pickle"

export ENHANCE_RUN_ID="skipped"

if [ "$MOVE_ON" = true ]; then
source ./BB_Stages_bash/STG2_DVECTORS_ENHANCER.sh
fi

#### Stage 3 Unsupervised Method
export current_stg3="${ROOT_PATH}/${DATASET_NAME}/STG_3/STG3_${EXP_NAME}-${VAD_NAME}-${FEAT_NAME}-${METHOD_NAME}"
export STG3_MERGED_WAVS="${current_stg3}/merged_wavs"
export STG3_FINAL_CSV="${current_stg3}/final_csv"

export pca_elem="0"

export min_cluster_size="25"
export hdb_mode="eom"
export min_samples="5"

export RUN_PARAMS="pca${pca_elem}_mcs${min_cluster_size}_ms${min_samples}_${hdb_mode}"

cd $SRC_PATH
if [ "$MOVE_ON" = true ]; then
source ./BB_Stages_bash/STG3_META_HDB.sh
fi


# #### Stage 4 Metrics
# export STG1_GT_CSV="${ROOT_PATH}/${DATASET_NAME}/GT_final/"
# export STG4_METRICS="${current_stg3}/metrics"
# export STG4_METRIC_RUNNAME="${DATASET_NAME}_${VAD_NAME}_${FEAT_NAME}_${METHOD_NAME}"

# export pred_suffix_added="pred"
# export pred_ext="csv"


# cd $SRC_PATH
# if [ "$MOVE_ON" = true ]; then
#     source STG4_ENTROPY.sh
# fi

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