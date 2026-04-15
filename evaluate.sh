#!/bin/bash

TASKS=(
    "lift_lid"
    "open_box"
    "phone_on_base"
    "plate_out"
    "close_microwave"
    "push_button"
    "slide_block"
    "close_box"
    "basketball"
    "buzz"
    "toilet_seat_down"
    "toilet_seat_up"
    "toilet_roll_off"
    "open_microwave"
    "lamp_on"
    "umbrella_out"
    "put_rubbish"
)

echo "Starting batch execution of all IGA tasks..."

for task in "${TASKS[@]}"
do
    echo "==================================================="
    echo "Task: $task"
    echo "==================================================="
    
    # xvfb-run
    xvfb-run -a -s "-screen 0 640x480x16 +extension GLX +render -noreset -nocursor" \
        python run_iga_sim.py --task "$task" --num_demos 1 --num_rollouts 10
    
    STATUS=$?
    
    if [ $STATUS -eq 0 ]; then
        echo "[FINISH] Task '$task' finished."
    else
        echo "[ERROR] Task '$task' exited with code $STATUS. Continuing to next task..."
    fi
    
    echo ""
done

echo "All tasks processed."
