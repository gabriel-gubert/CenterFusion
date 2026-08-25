#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

cd ../src

MODES=("primary_as_storage" "all_as_storage" "secondary_as_storage")

start() {
    local N=$1
    local Es=$2
    local MODE=$3

    local TITLE_MODE=$(echo "$MODE" | sed -E 's/_/-/g; s/\b[a-z]/\U&/g; s/-As-/-as-/g')

    local LABEL="Posit-${N}-${Es}-${TITLE_MODE}"

    echo "--------------------------------------------------------"
    echo "Evaluating Posit ($N, $Es)-Quantized CenterFusion..."
    echo "Mode: $MODE (Formatted for Label as: $TITLE_MODE)"
    echo "Label: $LABEL"
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
        --inference_num_workers 4

    sleep 1
}

for MODE in "${MODES[@]}"; do

    for N in {2..32}; do
        start "$N" 2 "$MODE"
    done

    start 16 1 "$MODE"

    start 8 0 "$MODE"

done