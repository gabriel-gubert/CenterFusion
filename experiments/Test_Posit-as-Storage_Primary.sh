export CUDA_VISIBLE_DEVICES=0

cd ../src

N=$2
Es=$3
LABEL="Posit-${N}-${Es}-Primary-as-Storage"

echo "Evaluating Posit ($N, $Es)-Quantized CenterFusion..."
python test.py ddd \
	--dataset nuscenes \
	--exp_id $LABEL \
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
	--quantize_heads primary_as_storage \
	--N $N \
	--Es $Es \
	--inference_num_workers 4