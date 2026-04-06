#!/bin/bash

# 定义要测试的所有任务列表
TASKS=(
    "lift_lid"
    "phone_on_base"
    "open_box"
    "slide_block"
    "close_box"
    "basketball"
    "buzz"
    "close_microwave"
    "plate_out"
    "toilet_seat_down"
    "toilet_seat_up"
    "toilet_roll_off"
    "open_microwave"
    "lamp_on"
    "umbrella_out"
    "push_button"
    "put_rubbish"
)

echo "Starting batch execution of all IGA tasks..."

for task in "${TASKS[@]}"
do
    echo "==================================================="
    echo "Task: $task"
    echo "==================================================="
    
    # 结合 xvfb-run 以免在无头环境下报错退出
    xvfb-run -a -s "-screen 0 640x480x16 +extension GLX +render -noreset -nocursor" \
        python run_iga_sim.py --task "$task"
    
    # 捕获退出状态码
    STATUS=$?
    
    if [ $STATUS -eq 0 ]; then
        echo "[SUCCESS] Task '$task' finished successfully."
    else
        echo "[ERROR] Task '$task' exited with code $STATUS. Continuing to next task..."
    fi
    
    echo ""
done

echo "All tasks processed."
