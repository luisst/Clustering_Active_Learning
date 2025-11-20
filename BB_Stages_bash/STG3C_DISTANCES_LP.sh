#! /bin/bash

export MOVE_ON=true

echo -e "\n"
echo -e "========================================================================"
echo -e "STAGE 4C: DISTANCE-BASED LABEL PROPAGATION (DIRECT FROM STAGE 2)"
echo -e "========================================================================"
echo -e "This stage bypasses all clustering (Stage 3) and goes directly from"
echo -e "Stage 2 D-vectors to Label Propagation using distance-based affinities."
echo -e ""
echo -e "Pipeline steps:"
echo -e "  1. Smart Sample Selection (Active Learning)"
echo -e "  2. Manual Labeling (User fills in speaker names)"
echo -e "  3. Distance-Based Label Propagation"
echo -e "========================================================================\n"

# Smart Sample Selection Parameters
export SMART_SELECTION_METHOD="hybrid"  # Options: hybrid, uncertainty, kmeans
export N_SAMPLES_TO_LABEL="20"          # Number of samples to select for labeling
export N_CLUSTERS_ESTIMATE="10"         # Estimated number of speakers/clusters
export SELECTION_SEED="42"              # Random seed for reproducibility
export REDUCTION_METHOD="pca"           # Options: pca, umap (dimensionality reduction)

# Distance-based Label Propagation Parameters
export DISTANCE_METRIC="cosine"  # Options: euclidean, manhattan, cosine
export AFFINITY_METHOD="rbf"     # Options: rbf, inverse
export SIGMA_VALUE=""            # Leave empty for auto (median distance)
export KNN_SPARSIFY=""           # Leave empty for dense matrix, or set to integer (e.g., 15)
export ADD_MST_FLAG=true         # Add MST for connectivity

export LP_ALPHA="0.5"            # Propagation strength
export LP_MAX_ITER="100"         # Maximum iterations
export LP_TOL="0.000001"         # Convergence tolerance
export ANCHOR_STRENGTH="0.9"     # Manual label anchoring strength (higher = stronger)
export TSNE_PERPLEXITY="30"      # t-SNE perplexity for visualization

# UMAP parameters for visualization
export UMAP_N_COMPONENTS="20"    # UMAP dimensions before t-SNE
export UMAP_N_NEIGHBORS="15"     # UMAP n_neighbors parameter
export UMAP_MIN_DIST="0.1"       # UMAP min_dist parameter

# Verify/create output folders
python3 ${SRC_PATH}/folder_create.py $STG4_LP_OUTPUT_FOLDER

conda activate metaSR3

## Stage 4c-1: Smart Sample Selection (if human labels don't exist)
echo -e "\n\t>>>>> Stage 4c-1: Smart Sample Selection\n"

# Define path for sample selection outputs (CSVs organized by long wav)
export SAMPLES_OUTPUT_FOLDER="${STG4_LP_OUTPUT_FOLDER}/selected_samples"

# Check if human labels JSON already exists
if [ -f "$STG4_HUMAN_LABELS_JSON" ]; then
    echo -e "  ✓ Human labels JSON already exists: $STG4_HUMAN_LABELS_JSON"
    echo -e "  Skipping smart sample selection\n"
else
    echo -e "  Human labels JSON not found, running smart sample selection..."
    echo -e "  - D-vectors pickle: $STG2_FEATS_ENHANCED"
    echo -e "  - Selection method: $SMART_SELECTION_METHOD"
    echo -e "  - Reduction method: $REDUCTION_METHOD"
    echo -e "  - Samples to select: $N_SAMPLES_TO_LABEL"
    echo -e "  - Estimated clusters: $N_CLUSTERS_ESTIMATE"
    echo -e "  - Output folder: $SAMPLES_OUTPUT_FOLDER\n"

    # Build smart selection command
    SELECTION_CMD="python3 ${SRC_PATH}/04_Active_learning_loop/stg4c_smart_sample_selection.py"
    SELECTION_CMD="$SELECTION_CMD --dvectors_pickle $STG2_FEATS_ENHANCED"
    SELECTION_CMD="$SELECTION_CMD --output_folder $SAMPLES_OUTPUT_FOLDER"
    SELECTION_CMD="$SELECTION_CMD --n_samples $N_SAMPLES_TO_LABEL"
    SELECTION_CMD="$SELECTION_CMD --method $SMART_SELECTION_METHOD"
    SELECTION_CMD="$SELECTION_CMD --n_clusters $N_CLUSTERS_ESTIMATE"
    SELECTION_CMD="$SELECTION_CMD --seed $SELECTION_SEED"
    SELECTION_CMD="$SELECTION_CMD --reduction_method $REDUCTION_METHOD"

    # Execute smart selection
    echo -e "Executing smart selection command:\n"
    echo -e "$SELECTION_CMD\n"
    eval $SELECTION_CMD

    # Check if selection was successful
    if [ $? -ne 0 ]; then
        export MOVE_ON=false
        echo -e "\n\t ✗ ERROR: Smart sample selection failed"
        return 1
    fi

    # Provide instructions for manual labeling
    echo -e "\n========================================================================"
    echo -e "MANUAL LABELING REQUIRED"
    echo -e "========================================================================"
    echo -e "Smart sample selection completed successfully!"
    echo -e ""
    echo -e "CSV files created (one per long wav file):"
    echo -e "  Location: ${SAMPLES_OUTPUT_FOLDER}"
    echo -e ""
    echo -e "Next steps:"
    echo -e "  1. Review CSV files in: ${SAMPLES_OUTPUT_FOLDER}"
    echo -e "     Each CSV contains selected samples from one long wav"
    echo -e "     Format: filename,cluster_id,start_time,end_time (comma-separated, no header)"
    echo -e ""
    echo -e "  2. Listen to the audio files"
    echo -e "     Audio chunks location: ${STG1_FILTERED_CHUNKS_WAVS}"
    echo -e "     Audio file: \${STG1_FILTERED_CHUNKS_WAVS}/\${filename}.wav"
    echo -e ""
    echo -e "  3. Use the webapp to label speakers and generate human_labels.json"
    echo -e "     Or manually create: ${STG4_HUMAN_LABELS_JSON}"
    echo -e ""
    echo -e "  4. Re-run this pipeline to proceed with Label Propagation"
    echo -e "========================================================================\n"

    export MOVE_ON=false
    return 0
fi

## Stage 4c-2: Distance-Based Label Propagation (from Stage 2 D-vectors)
echo -e "\n\t>>>>> Stage 4c-2: Distance-Based Label Propagation\n"
echo -e "  - D-vectors pickle: $STG2_FEATS_ENHANCED"
echo -e "  - Human labels JSON: $STG4_HUMAN_LABELS_JSON"
echo -e "  - Output folder: $STG4_LP_OUTPUT_FOLDER"
echo -e "  - Distance metric: $DISTANCE_METRIC"
echo -e "  - Affinity method: $AFFINITY_METHOD"
echo -e "  - Anchor strength: $ANCHOR_STRENGTH"
echo -e "  - UMAP dimensions: $UMAP_N_COMPONENTS\n"

# Check if human labels JSON exists
if [ ! -f "$STG4_HUMAN_LABELS_JSON" ]; then
    echo -e "\n\t ✗ ERROR: Human labels JSON not found: $STG4_HUMAN_LABELS_JSON"
    echo -e "\tThis should not happen if you followed the manual labeling steps."
    echo -e "\tPlease check the file path and try again.\n"
    export MOVE_ON=false
    return 1
fi

# Build command
LP_CMD="python3 ${SRC_PATH}/04_Active_learning_loop/stg4C_distances_LP.py"
LP_CMD="$LP_CMD --dvectors_pickle $STG2_FEATS_ENHANCED"
LP_CMD="$LP_CMD --human_labels_json $STG4_HUMAN_LABELS_JSON"
LP_CMD="$LP_CMD --output_folder $STG4_LP_OUTPUT_FOLDER"
LP_CMD="$LP_CMD --run_id $STG4_RUN_ID"
LP_CMD="$LP_CMD --distance_metric $DISTANCE_METRIC"
LP_CMD="$LP_CMD --affinity_method $AFFINITY_METHOD"
LP_CMD="$LP_CMD --alpha $LP_ALPHA"
LP_CMD="$LP_CMD --max_iter $LP_MAX_ITER"
LP_CMD="$LP_CMD --tol $LP_TOL"
LP_CMD="$LP_CMD --anchor_strength $ANCHOR_STRENGTH"
LP_CMD="$LP_CMD --tsne_perplexity $TSNE_PERPLEXITY"
LP_CMD="$LP_CMD --umap_n_components $UMAP_N_COMPONENTS"
LP_CMD="$LP_CMD --umap_n_neighbors $UMAP_N_NEIGHBORS"
LP_CMD="$LP_CMD --umap_min_dist $UMAP_MIN_DIST"

# Add optional parameters
if [ -n "$SIGMA_VALUE" ]; then
    LP_CMD="$LP_CMD --sigma $SIGMA_VALUE"
fi

if [ -n "$KNN_SPARSIFY" ]; then
    LP_CMD="$LP_CMD --knn_sparsify $KNN_SPARSIFY"
fi

if [ "$ADD_MST_FLAG" = "true" ]; then
    LP_CMD="$LP_CMD --add_mst"
fi

# Execute label propagation
echo -e "Executing LP command:\n"
echo -e "$LP_CMD\n"
eval $LP_CMD

# Check if the Python script was successful
if [ $? -ne 0 ]; then
    export MOVE_ON=false
    echo -e "\n\t ✗ ERROR: Stage 4c failed"
    echo "Move on: $MOVE_ON"
    return 1
else
    echo -e "\n\t ✓ Stage 4c completed successfully"
    echo -e "\tLP Results CSV: $STG4_LP_RESULTS_CSV"
    echo -e "\tVisualization: ${STG4_LP_OUTPUT_FOLDER}/${STG4_RUN_ID}_visualization.png"
fi

# Unset stage-specific variables
unset SMART_SELECTION_METHOD
unset N_SAMPLES_TO_LABEL
unset N_CLUSTERS_ESTIMATE
unset SELECTION_SEED
unset REDUCTION_METHOD
unset SAMPLES_OUTPUT_FOLDER
unset SELECTION_CMD
unset DISTANCE_METRIC
unset AFFINITY_METHOD
unset SIGMA_VALUE
unset KNN_SPARSIFY
unset ADD_MST_FLAG
unset LP_ALPHA
unset LP_MAX_ITER
unset LP_TOL
unset ANCHOR_STRENGTH
unset TSNE_PERPLEXITY
unset UMAP_N_COMPONENTS
unset UMAP_N_NEIGHBORS
unset UMAP_MIN_DIST
unset LP_CMD
