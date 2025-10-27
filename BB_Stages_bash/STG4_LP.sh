#! /bin/bash

export MOVE_ON=true
python3 ${SRC_PATH}/folder_verify.py $STG4_HUMAN


# ## Run the HDBSCAN prediction
# echo -e "\n\t>>>>> Stg4a Convert to webApp: $STG4_AL_INPUT\n"
# python3 ${SRC_PATH}/04_Active_learning_loop/stg4a_generate_ALinputs.py --stg3_al_input $STG3_AL_INPUT\
#     --stg4_al_folder $current_stg4

# # Check if the Python script was successful
# if [ $? -ne 0 ]; then
#     export MOVE_ON=false
#     echo "Move on: $MOVE_ON"
#     return 1
# fi


echo -e "\n\t>>>>> Stg4b Marker webapp: $STG4_AL_INPUT\n"
python3 ${SRC_PATH}/04_Active_learning_loop/stg4b_marker_webapp.py --stg1_mp4_candidate $STG1_MP4_FOLDER\
    --stg4_al_folder $current_stg4


