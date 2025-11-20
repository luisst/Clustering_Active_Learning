#! /bin/bash
echo -e "Alternative Pipeline: Distance-Based Label Propagation\n"
export MOVE_ON=true
#### Root folder from dropbox
export ROOT_PATH="/home/luis/Dropbox/DATASETS_AUDIO/Unsupervised_Pipeline"
export SRC_PATH=$(pwd)

export EXP_NAME="EXP010"
# export DATASET_NAME="TTS4_easy"
export DATASET_NAME="TestAO-IrmaAlt"


export VAD_NAME="SHAS"
export FEAT_NAME="DV"
export METHOD_NAME="hdb"
export AZURE_FLAG=false

## Optional methods
export DOUBLE_TALK_FLAG=true
export SILENT_DET_FLAG=true
export ENHANCE_FLAG=false

## If GT_CSV_FOLDER is provided, predict_only will be set to false
export GT_CSV_FOLDER="${ROOT_PATH}/${DATASET_NAME}/GT_final/filtered_GT"
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
export STG2_ENH_PRETRAINED="${SRC_PATH}/pre-trained/model_S1_256_hard_betaNO4_mask00_lr-5_ep180_73.pth"
export PRETRAINED_DVECTOR_PATH="${SRC_PATH}/pre-trained/checkpoint_100_original_5994.pth"

## Segmentation Parameters
export seg_ln="1.0"
export step_size="0.2"
export gap_size="0.2"
export consc_th="1"
export min_overlap_percentage="0.8"
export DT_THRESHOLD="0.8"

export STG1_MP4_FOLDER="${ROOT_PATH}/${DATASET_NAME}/input_mp4s"

#### Stage 1 VAD
export current_stg1="${ROOT_PATH}/${DATASET_NAME}/STG_1/STG1_${VAD_NAME}"
export STG1_WAVS="${ROOT_PATH}/${DATASET_NAME}/input_wavs/"
export STG1_FILTERED_CHUNKS_WAVS="${current_stg1}/wav_chunks_filtered"

# Stage 1f: Speaker JSON variables
export STG1_SPEAKERS_DATABASE_JSON="${SRC_PATH}/04_Active_learning_loop/images_june4.json"
export STG1_SPEAKERS_JSON="${STG1_WAVS}/speakers_info.json"
export STG1_GT_FOLDER="${GT_CSV_FOLDER}"
export SPEAKERS_JSON_FILENAME="speakers_by_basename.json"

### NOTE: Stage 1 is commented out - assumes Stage 1 outputs already exist
# source ./BB_Stages_bash/STG1_SHAS.sh

#### Stage 2 Feature Extraction
export current_stg2="${ROOT_PATH}/${DATASET_NAME}/STG_2/STG2_${EXP_NAME}-${VAD_NAME}-${FEAT_NAME}"
export STG2_FEATS_PICKLE="${current_stg2}/${DATASET_NAME}_${VAD_NAME}_${FEAT_NAME}_feats.pickle"
export STG2_FEATS_ENHANCED="${current_stg2}/${DATASET_NAME}_${VAD_NAME}_${FEAT_NAME}_featsEN.pickle"
export ENHANCE_RUN_ID="skipped"

# ### NOTE: Stage 2 is commented out - assumes Stage 2 outputs already exist
# if [ "$MOVE_ON" = "true" ]; then
#     source ./BB_Stages_bash/STG2_DVECTORS_ENHANCER.sh
# fi

#### NOTE: Stage 3 (Clustering) is completely skipped in this pipeline
# This distance-based LP approach goes directly from Stage 2 D-vectors to LP

#### Stage 4: Distance-Based Label Propagation (Direct from Stage 2)
export LP_METHOD_NAME="LP_DIST"
export current_stg4="${ROOT_PATH}/${DATASET_NAME}/STG_4/STG4_${LP_METHOD_NAME}"
export STG4_LP_OUTPUT_FOLDER="${current_stg4}/lp_results"
export STG4_RUN_ID="RUN_DIST001"
export STG4_LP_RESULTS_CSV="${STG4_LP_OUTPUT_FOLDER}/${STG4_RUN_ID}_lp_results.csv"

# Human labels JSON - YOU MUST CREATE THIS FILE
# Format: {"sample_id_1": "Speaker_A", "sample_id_2": "Speaker_B", ...}
export STG4_HUMAN_LABELS_JSON="${current_stg4}/human_labels.json"

echo -e "\n"
echo -e "========================================================================"
echo -e "DISTANCE-BASED LABEL PROPAGATION PIPELINE WITH SMART SELECTION"
echo -e "========================================================================"
echo -e "This pipeline skips all clustering stages and performs Label"
echo -e "Propagation directly from Stage 2 D-vectors using distance metrics."
echo -e ""
echo -e "Pipeline flow:"
echo -e "  Stage 1 (VAD) -> Stage 2 (D-vectors) -> Smart Selection ->"
echo -e "  -> Manual Labeling -> Distance-Based LP"
echo -e ""
echo -e "Features:"
echo -e "  - Smart sample selection using active learning (SVM + K-Means)"
echo -e "  - Intelligently selects most informative samples to label"
echo -e "  - Minimal manual labeling effort (20 samples out of 300)"
echo -e "  - Direct distance-based LP without clustering"
echo -e ""
echo -e "Requirements:"
echo -e "  1. Stage 2 D-vectors pickle must exist: $STG2_FEATS_ENHANCED"
echo -e ""
echo -e "On first run:"
echo -e "  - Smart selection will identify 20 samples to label"
echo -e "  - Pipeline will pause for manual labeling"
echo -e "  - Follow instructions to fill in speaker names"
echo -e "  - Re-run pipeline to complete LP"
echo -e "========================================================================"
echo -e "\n"

cd $SRC_PATH
if [ "$MOVE_ON" = "true" ]; then
    source ./BB_Stages_bash/STG3C_DISTANCES_LP.sh
fi

#### Stage 5 Metrics Comparison (Optional)
export STG5_AZURE_FOLDER="${ROOT_PATH}/${DATASET_NAME}/azure_diarization_output"
export current_stg5="${ROOT_PATH}/${DATASET_NAME}/STG_5"
export STG5_LP_RESULTS="${current_stg5}/LP_DIST_metrics"
export STG5_AZURE_RESULTS="${current_stg5}/Azure_metrics"

# Uncomment to run Stage 5 metrics:
# cd $SRC_PATH
# if [ "$MOVE_ON" = "true" ]; then
#     source ./BB_Stages_bash/STG5_METRICS.sh
# fi

#### Get back to where you once belonged
conda activate
cd $SRC_PATH

# # Unset all the variables
# unset MOVE_ON
# unset ROOT_PATH
# unset SRC_PATH
# unset EXP_NAME
# unset DATASET_NAME
# unset VAD_NAME
# unset FEAT_NAME
# unset METHOD_NAME
# unset AZURE_FLAG
# unset DOUBLE_TALK_FLAG
# unset SILENT_DET_FLAG
# unset ENHANCE_FLAG
# unset GT_CSV_FOLDER
# unset PREDICT_ONLY
# unset USE_PKL_LABEL
# unset VAD_LOCATION
# unset STG1_DT_PRETRAINED
# unset STG1_VAD_PRETRAINED
# unset STG2_ENH_PRETRAINED
# unset PRETRAINED_DVECTOR_PATH
# unset seg_ln
# unset step_size
# unset gap_size
# unset consc_th
# unset min_overlap_percentage
# unset DT_THRESHOLD
# unset STG1_MP4_FOLDER
# unset current_stg1
# unset STG1_WAVS
# unset STG1_FILTERED_CHUNKS_WAVS
# unset STG1_SPEAKERS_DATABASE_JSON
# unset STG1_SPEAKERS_JSON
# unset STG1_GT_FOLDER
# unset SPEAKERS_JSON_FILENAME
# unset current_stg2
# unset STG2_FEATS_PICKLE
# unset STG2_FEATS_ENHANCED
# unset ENHANCE_RUN_ID
# unset LP_METHOD_NAME
# unset current_stg4
# unset STG4_LP_OUTPUT_FOLDER
# unset STG4_RUN_ID
# unset STG4_LP_RESULTS_CSV
# unset STG4_HUMAN_LABELS_JSON
# unset STG5_AZURE_FOLDER
# unset current_stg5
# unset STG5_LP_RESULTS
# unset STG5_AZURE_RESULTS
