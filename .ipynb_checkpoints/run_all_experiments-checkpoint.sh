#!/bin/bash
# run_all_experiments.sh

# Usage: chmod +x run_all_experiments.sh && ./run_all_experiments.sh


echo "Starting all experiments..."

/usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization none --save_path ckpts/tas_none.pth 
/usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization minmax_global --save_path ckpts/tas_minmax_global.pth 
/usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization minmax_pixel --save_path ckpts/tas_minmax_pixel.pth 
/usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization zscore_global --save_path ckpts/tas_zscore_global.pth 
/usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization zscore_pixel --save_path ckpts/tas_zscore_pixel.pth 
/usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization instance_zscore --save_path ckpts/tas_instance_zscore.pth 
/usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization instance_minmax --save_path ckpts/tas_instance_minmax.pth 
/usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization none --save_path ckpts/pr_none.pth 
/usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization minmax_global --save_path ckpts/pr_minmax_global.pth 
/usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization minmax_pixel --save_path ckpts/pr_minmax_pixel.pth 
/usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization zscore_global --save_path ckpts/pr_zscore_global.pth 
/usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization zscore_pixel --save_path ckpts/pr_zscore_pixel.pth 
/usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization instance_zscore --save_path ckpts/pr_instance_zscore.pth 
/usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization instance_minmax --save_path ckpts/pr_instance_minmax.pth 
/usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization none --save_path ckpts/hurs_none.pth 
/usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization minmax_global --save_path ckpts/hurs_minmax_global.pth 
/usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization minmax_pixel --save_path ckpts/hurs_minmax_pixel.pth 
/usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization zscore_global --save_path ckpts/hurs_zscore_global.pth 
/usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization zscore_pixel --save_path ckpts/hurs_zscore_pixel.pth 
/usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization instance_zscore --save_path ckpts/hurs_instance_zscore.pth 
/usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization instance_minmax --save_path ckpts/hurs_instance_minmax.pth 
/usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization none --save_path ckpts/sfcWind_none.pth 
/usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization minmax_global --save_path ckpts/sfcWind_minmax_global.pth 
/usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization minmax_pixel --save_path ckpts/sfcWind_minmax_pixel.pth 
/usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization zscore_global --save_path ckpts/sfcWind_zscore_global.pth 
/usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization zscore_pixel --save_path ckpts/sfcWind_zscore_pixel.pth 
/usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization instance_zscore --save_path ckpts/sfcWind_instance_zscore.pth 
/usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization instance_minmax --save_path ckpts/sfcWind_instance_minmax.pth

echo "All experiments completed!"