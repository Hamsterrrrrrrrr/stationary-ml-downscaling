#!/bin/bash
# run_residual_experiments.sh

# chmod +x run_residual_experiments.sh
# ./run_residual_experiments.sh


# Precipitation experiments
echo "Starting precipitation experiments..."
/usr/local/anaconda3/bin/python main_residual.py --variables pr_hr --input_type raw --save_path ckpts/pr_lr_to_residual.pth
/usr/local/anaconda3/bin/python main_residual.py --variables pr_hr --input_type detrend_gma --save_path ckpts/pr_lr_gma_to_residual.pth
/usr/local/anaconda3/bin/python main_residual.py --variables pr_hr --input_type detrend_grid --save_path ckpts/pr_lr_grid_to_residual.pth
/usr/local/anaconda3/bin/python main_residual.py --variables pr_hr --input_type detrend_gmt --save_path ckpts/pr_lr_gmt_to_residual.pth

# Temperature experiments
echo "Starting temperature experiments..."
/usr/local/anaconda3/bin/python main_residual.py --variables tas_hr --input_type raw --save_path ckpts/tas_lr_to_residual.pth
/usr/local/anaconda3/bin/python main_residual.py --variables tas_hr --input_type detrend_gma --save_path ckpts/tas_lr_gma_to_residual.pth
/usr/local/anaconda3/bin/python main_residual.py --variables tas_hr --input_type detrend_grid --save_path ckpts/tas_lr_grid_to_residual.pth
/usr/local/anaconda3/bin/python main_residual.py --variables tas_hr --input_type detrend_gmt --save_path ckpts/tas_lr_gmt_to_residual.pth

echo "All experiments completed!"