export CUDA_VISIBLE_DEVICES=0
 
cd ../src

N=$1
Es=$2
LABEL="KV260-Posit-${N}-${Es}-All"

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
	--quantize_heads all \
	--N $N \
	--Es $Es \
	--qdevice fpga \
	--fpga_host 10.116.35.155 \
	--fpga_port 8080 \
	--fpga_conf "KV260 8 8 $N $Es 128 512" \
	--inference_num_workers 4