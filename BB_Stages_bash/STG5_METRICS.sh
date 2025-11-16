#! /bin/bash

export MOVE_ON=true
export SKIP_STG5a=false
export SKIP_STG5b=false

# Verify/create Stage 5 folders
python3 ${SRC_PATH}/folder_verify.py $STG5_LP_RESULTS
python3 ${SRC_PATH}/folder_verify.py $STG5_AZURE_RESULTS

# Check if LP results CSV exists
if [ ! -f "$STG4_LP_RESULTS_CSV" ]; then
    export SKIP_STG5a=true
    echo -e "\n\t LP results CSV not found: $STG4_LP_RESULTS_CSV"
    echo -e "\tSkipping Stage 5a: LP Metrics\n"
fi

# Check if Azure folder exists
if [ ! -d "$STG5_AZURE_FOLDER" ]; then
    export SKIP_STG5b=true
    echo -e "\n\t Azure diarization folder not found: $STG5_AZURE_FOLDER"
    echo -e "\tSkipping Stage 5b: Azure Metrics\n"
fi

conda activate metaSR3

echo -e "\n=================================================="
echo -e "STAGE 5: METRICS COMPARISON"
echo -e "==================================================\n"

## Stage 5a: Calculate LP (Label Propagation) Metrics
echo -e "\n\t>>>>> Stage 5a: LP Metrics with Timing and GT Comparison\n"
if [ "$SKIP_STG5a" != "true" ]; then
    echo -e "  - LP Results CSV: $STG4_LP_RESULTS_CSV"
    echo -e "  - Merged HDF5: $STG3_MERGED_H5"
    echo -e "  - GT Folder: $GT_CSV_FOLDER"
    echo -e "  - Output Folder: $STG5_LP_RESULTS\n"

    python3 ${SRC_PATH}/05_Pipeline_metrics/Stg5a_create_csv_from_merged.py \
        --lp_results_csv $STG4_LP_RESULTS_CSV \
        --merged_dataset_h5 $STG3_MERGED_H5 \
        --gt_csv_folder $GT_CSV_FOLDER \
        --output_folder $STG5_LP_RESULTS \
        --min_overlap_pct 50

    # Check if the Python script was successful
    if [ $? -ne 0 ]; then
        export MOVE_ON=false
        echo -e "\n\t Stage 5a failed"
        echo "Move on: $MOVE_ON"
        return 1
    else
        echo -e "\n\t Stage 5a completed successfully"
        echo -e "\tResults saved to: $STG5_LP_RESULTS"
    fi
else
    echo -e "\n\t>>>>> Stage 5a: LP Metrics skipped\n"
fi


## Stage 5b: Calculate Azure Metrics
echo -e "\n\t>>>>> Stage 5b: Azure Speech Service Diarization Metrics\n"
if [ "$SKIP_STG5b" != "true" ]; then
    echo -e "  - Azure Folder: $STG5_AZURE_FOLDER"
    echo -e "  - GT Folder: $GT_CSV_FOLDER"
    echo -e "  - Output Folder: $STG5_AZURE_RESULTS\n"

    python3 ${SRC_PATH}/06_Metric_competition/Azure_metrics.py \
        --azure_folder $STG5_AZURE_FOLDER \
        --gt_csv_folder $GT_CSV_FOLDER \
        --output_folder $STG5_AZURE_RESULTS \
        --min_overlap_pct 50

    # Check if the Python script was successful
    if [ $? -ne 0 ]; then
        export MOVE_ON=false
        echo -e "\n\t Stage 5b failed"
        echo "Move on: $MOVE_ON"
        return 1
    else
        echo -e "\n\t Stage 5b completed successfully"
        echo -e "\tResults saved to: $STG5_AZURE_RESULTS"
    fi
else
    echo -e "\n\t>>>>> Stage 5b: Azure Metrics skipped\n"
fi


## Summary
echo -e "\n=================================================="
echo -e "STAGE 5 METRICS COMPARISON COMPLETED"
echo -e "==================================================\n"

if [ "$SKIP_STG5a" != "true" ]; then
    echo -e "LP Metrics Report: $STG5_LP_RESULTS/metrics_report.txt"
    echo -e "LP Confusion Matrix: $STG5_LP_RESULTS/confusion_matrix.png"
    echo -e "LP Metrics Summary: $STG5_LP_RESULTS/metrics_summary.png"
fi

if [ "$SKIP_STG5b" != "true" ]; then
    echo -e "Azure Metrics Report: $STG5_AZURE_RESULTS/azure_metrics_report.txt"
    echo -e "Azure Confusion Matrix: $STG5_AZURE_RESULTS/confusion_matrix_overall.png"
    echo -e "Azure Per-File Comparison: $STG5_AZURE_RESULTS/per_file_metrics_comparison.png"
fi

echo -e "\n"

# Unset all stage 5 variables
unset SKIP_STG5a
unset SKIP_STG5b
