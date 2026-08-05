#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
CURRENT_WORKING_DIR=$(pwd)
cd ~/CenterFusion/src

FPGA_HOST="10.116.35.155"
FPGA_PORT="8080"
FPGA_CONF_TEMPLATE="KV260 8 8 %N% %Es% 128 512"

MODES=("primary" "all" "secondary")

run_evaluation() {
    local N=$1
    local Es=$2
    local MODE=$3
    
    local TITLE_MODE=$(echo "$MODE" | sed -E 's/(^|_)([a-z])/\1\U\2/g')
    
    local TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    local LABEL="Posit_${N}_${Es}_${TITLE_MODE}_${TIMESTAMP}"
    
    local FPGA_CONF="${FPGA_CONF_TEMPLATE//%N%/$N}"
    FPGA_CONF="${FPGA_CONF//%Es%/$Es}"

    echo "--------------------------------------------------------"
    echo "Evaluating Posit ($N, $Es)-Quantized CenterFusion..."
    echo "Mode: $MODE (Formatted for Label as: $TITLE_MODE)"
    echo "Label: $LABEL"
    echo "FPGA Conf: $FPGA_CONF"
    echo "--------------------------------------------------------"

    python test.py ddd \
        --dataset nuscenes \
        --exp_id "$LABEL" \
        --load_model ../models/centerfusion_e60.pth \
        --debug 4 \
        --no_pause \
        --gpus 0 \
        --run_dataset_eval \
        --input_h 448 \
        --input_w 800 \
        --flip_test \
        --save_results \
        --nuscenes_att \
        --velocity \
        --pointcloud \
        --val_split mini_val \
        --max_pc_dist 60.0 \
        --radar_sweeps 3 \
        --pc_z_offset -0.0 \
        --eval_render_curves \
        --show_velocity \
        --quantize_heads "$MODE" \
        --N "$N" \
        --Es "$Es" \
        --qdevice fpga \
        --fpga_host "$FPGA_HOST" \
        --fpga_port "$FPGA_PORT" \
        --fpga_conf "$FPGA_CONF" \
        --inference_num_workers 4
        
    sleep 1
}

for MODE in "${MODES[@]}"; do
    for N in {6..8}; do
        run_evaluation "$N" 2 "$MODE"
    done
done

cd "$CURRENT_WORKING_DIR"