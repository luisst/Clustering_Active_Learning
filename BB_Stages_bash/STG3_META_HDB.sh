#! /bin/bash
export SKIP_STG3A=false
export SKIP_metrics=false
export SKIP_merged=false
export SKIP_STG3e=false

export STG3_HDBSCAN_PRED_OUTPUT="${current_stg3}/HDBSCAN_pred_output"
export STG3_CLUSTERING_PICKLE="${current_stg3}/clustering_data.pickle"
export STG3_REDUCED_FEATURES="${current_stg3}/reduced_features.pickle"
export STG3_CLUSTERING_LBLS="${current_stg3}/cluster_labels.pickle"

export STG3_CLUSTERING_METRICS="${current_stg3}/clustering_metrics"
export STG3_SEPARATED_WAVS="${current_stg3}/separated_wavs"
export STG3_OUTLIERS_WAVS="${current_stg3}/outliers_wavs"

export STG3_SEPARATED_MERGED_WAVS="${current_stg3}/separated_merged_wavs"

export RUN_ID="${EXP_NAME}_${DATASET_NAME}_${METHOD_NAME}"
export RUN_PARAMS="pca${pca_elem}_mcs${min_cluster_size}_ms${min_samples}_${hdb_mode}"


python3 ${SRC_PATH}/folder_verify.py $STG3_HDBSCAN_PRED_OUTPUT
if [ $? -eq 1 ]; then
    export SKIP_STG3A=true
fi

python3 ${SRC_PATH}/folder_verify.py $STG3_CLUSTERING_METRICS
if [ $? -eq 1 ]; then
    export SKIP_metrics=true
fi

python3 ${SRC_PATH}/folder_verify.py $STG3_MERGED_WAVS
if [ $? -eq 1 ]; then
    export SKIP_merged=true
fi

python3 ${SRC_PATH}/folder_verify.py $STG3_SEPARATED_WAVS
python3 ${SRC_PATH}/folder_verify.py $STG3_OUTLIERS_WAVS

python3 ${SRC_PATH}/folder_verify.py $STG3_FINAL_CSV
if [ $? -eq 1 ]; then
    export SKIP_STG3e=true
fi

python3 ${SRC_PATH}/folder_verify.py $STG3_SEPARATED_MERGED_WAVS

conda activate metaSR3

echo -e "prediction outputs will be saved in: $STG3_HDBSCAN_PRED_OUTPUT"

## Run the HDBSCAN prediction
echo -e "\n\t>>>>> Stg3A HDB-SCAN pred: $STG3_HDBSCAN_PRED_OUTPUT\n"
if [ "$SKIP_STG3A" != "true" ]; then
    python3 ${SRC_PATH}/03_Clustering_TDA/Stg3A_main_hdbscan_pred_output.py --input_feats_pickle $STG2_FEATS_ENHANCED\
    --output_pred_folder $STG3_HDBSCAN_PRED_OUTPUT\
    --run_params $RUN_PARAMS --exp_name $RUN_ID\
    --data_clusters_pickle $STG3_CLUSTERING_PICKLE --stg3_reduced_features $STG3_REDUCED_FEATURES\
    --stg3_pred_lbl $STG3_CLUSTERING_LBLS

    # Check if the Python script was successful
    if [ $? -ne 0 ]; then
        export MOVE_ON=false
        echo "Move on: $MOVE_ON"
        return 1
    fi
else
    echo -e "\n\t>>>>> Stage 3a: HDBSCAN prediction skipped\n"
fi

## Print Clustering metrics
echo -e "\n\t>>>>> Stg3c Clustering metrics $STG3_CLUSTERING_METRICS\n"
if [ "$SKIP_metrics" != "true" ]; then
    python3 ${SRC_PATH}/03_Clustering_TDA/Stg3c_internal_clusters.py --input_feats_pickle $STG2_FEATS_ENHANCED\
        --input_feats_reduced_pickle $STG3_REDUCED_FEATURES\
    --stg3_pred_lbl $STG3_CLUSTERING_LBLS --stg3_clustering_metrics $STG3_CLUSTERING_METRICS 

    # Check if the Python script was successful
    if [ $? -ne 0 ]; then
        export MOVE_ON=false
        echo "Move on: $MOVE_ON"
        return 1
    fi
else
    echo -e "\n\t>>>>> Stage 3c: Clustering metrics skipped\n"
fi


## Join the predictions chunks into a merged wav files
echo -e "\n\t>>>>> Stage 3d: Merged WAVs $STG3_MERGED_WAVS\n"
if [ "$SKIP_merged" != "true" ]; then
    python3 ${SRC_PATH}/03_Clustering_TDA/Stg3d_merge_wavs.py --stg1_long_wavs $STG1_WAVS\
    --stg3_pred_folders $STG3_HDBSCAN_PRED_OUTPUT --stg3_separated_wavs $STG3_SEPARATED_WAVS\
    --stg3_merged_wavs $STG3_MERGED_WAVS --stg3_outliers $STG3_OUTLIERS_WAVS\
    --data_clusters_pickle $STG3_CLUSTERING_PICKLE\
    --input_feats_pickle $STG2_FEATS_ENHANCED\
    --ln $seg_ln --st $step_size --gap $gap_size --consc_th $consc_th\
    --exp_name $EXP_NAME --AL_input_csv $STG3_AL_INPUT

    # Check if the Python script was successful
    if [ $? -ne 0 ]; then
        export MOVE_ON=false
        echo "Move on: $MOVE_ON"
        return 1
    fi
else
    echo -e "\n\t>>>>> Stage 3d: Merged WAVs skipped\n"
fi



## Create the final csv file
echo -e "\n\t>>>>> Stage 3e: Output Final CSV prediction $STG3_FINAL_CSV\n"
if [ "$SKIP_STG3e" != "true" ]; then
    python3 ${SRC_PATH}/03_Clustering_TDA/Stg3e_create_csv_from_merged.py --stg3_merged_wavs $STG3_MERGED_WAVS\
    --stg3_final_csv $STG3_FINAL_CSV --stg3_separated_merged_wavs $STG3_SEPARATED_MERGED_WAVS

    # Check if the Python script was successful
    if [ $? -ne 0 ]; then
        export MOVE_ON=false
        echo "Move on: $MOVE_ON"
        return 1
    fi
else
    echo -e "\n\t>>>>> Stage 3e: Final CSV prediction skipped\n"
fi

