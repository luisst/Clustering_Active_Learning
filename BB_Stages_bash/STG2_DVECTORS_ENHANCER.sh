#! /bin/bash
export SKIP_MFCC=false
export SKIP_FEATS=false
export SKIP_ENH=false

export STG2_MFCC_FILES="${current_stg2}/MFCC_files"


python3 ${SRC_PATH}/folder_verify.py $STG2_MFCC_FILES
if [ $? -eq 1 ]; then
    export SKIP_MFCC=true
fi

python3 ${SRC_PATH}/pickle_verify.py $STG2_FEATS_PICKLE
if [ $? -eq 1 ]; then
    export SKIP_FEATS=true
fi

python3 ${SRC_PATH}/pickle_verify.py $STG2_FEATS_ENHANCED
if [ $? -eq 1 ]; then
    export SKIP_ENH=true
fi

conda activate metaSR3


echo -e "\t>>>>> Stg2a Feature Extraction $STG2_MFCC_FILES"
if [ "$SKIP_MFCC" != "true" ]; then
    python3 ${SRC_PATH}/02_Feature_extraction/Stg2a_feature_extraction.py --wavs_folder $STG1_FILTERED_CHUNKS_WAVS\
    --output_feats_folder $STG2_MFCC_FILES

    # Check if the Python script was successful
    if [ $? -ne 0 ]; then
        export MOVE_ON=false
        echo "Move on: $MOVE_ON"
        return 1
    fi
else
    echo -e "\n\t>>>>> Stage 2a: MFCC extraction skipped\n"
fi


echo -e "\t>>>>> Stg2b D Vectors Inference $STG2_FEATS_PICKLE"
if [ "$SKIP_FEATS" != "true" ] && [ "$MOVE_ON" = "true" ]; then
    python3 ${SRC_PATH}/02_Feature_extraction/Stg2b_inference_dvectors.py --wavs_folder $STG1_FILTERED_CHUNKS_WAVS\
    --input_mfcc_folder $STG2_MFCC_FILES  --output_feats_pickle $STG2_FEATS_PICKLE --use_pkl_label $USE_PKL_LABEL\

    # Check if the Python script was successful
    if [ $? -ne 0 ]; then
        export MOVE_ON=false
        echo "Move on: $MOVE_ON"
        return 1
    fi
else
    echo -e "\n\t>>>>> Stage 2b: D Vectors inference skipped\n"
fi


echo -e "\t>>>>> Stage 2c: Running Inference Enhancer $STG2_FEATS_PICKLE"
if [ "$SKIP_ENH" != "true" ] && [ "$MOVE_ON" = "true" ]; then
    if [ "$ENHANCE_FLAG" = "true" ]; then
        python3 ${SRC_PATH}/02_Feature_extraction/Stg2c_enhance_feature.py --inference_feats_pickle $STG2_FEATS_PICKLE\
            --pretrained_model_path $STG2_ENH_PRETRAINED --enhanced_feats_pickle $STG2_FEATS_ENHANCED\
            --run_id $ENHANCE_RUN_ID

        # Check if the Python script was successful
        if [ $? -ne 0 ]; then
            export MOVE_ON=false
            echo "Move on: $MOVE_ON"
            return 1
        fi
    else
        echo -e "\n\t>>>>> Feature enhancement SKIPPED! \n"
        cp -r $STG2_FEATS_PICKLE $STG2_FEATS_ENHANCED
        # Check if the copy was successful
        if [ $? -ne 0 ]; then
            export MOVE_ON=false
            echo "Move on: $MOVE_ON"
            return 1
        fi
    fi

else
    echo -e "\n\t>>>>> Stage 2c: Feature enhancement skipped\n"
fi