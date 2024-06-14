export CUDA_VISIBLE_DEVICES=0,1,2,3
 
CURRENT_WORKING_DIR=$(pwd)
 
cd ~/CenterFusion/src

LABEL=$1
N=$2
Es=$3

if [ $N -eq 0 ] && [ $Es -eq 0 ]
then
	echo "Evaluating Non-Quantized CenterFusion..."
	## Perform detection and evaluation
	python test.py ddd \
		--dataset nuscenes \
		--exp_id $LABEL \
		--load_model ../models/centerfusion_e60.pth \
		--debug 4 \
		--no_pause \
		--gpus 0,1 \
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
		--show_velocity
else
	echo "Evaluating Posit ($N, $Es)-Quantized CenterFusion..."
	## Perform detection and evaluation
        python test.py ddd \
                --dataset nuscenes \
                --exp_id $LABEL \
                --load_model ../models/centerfusion_e60.pth \
                --debug 4 \
                --no_pause \
                --gpus 0,1 \
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
		--fpga_conf "KV260 8 8 8 2 128 512" \
                --inference_num_workers 4
fi

cd $CURRENT_WORKING_DIR
