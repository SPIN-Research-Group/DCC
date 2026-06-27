#!/bin/bash
## run kernel simulation and inference

if [ -d "./figures" ]; then
    rm -rf "./figures"
fi
if [ -d "./results" ]; then
    rm -rf "./results"
fi
mkdir "figures"
mkdir "results"


cd src

mkdir -p "temp/configs"
mkdir -p "temp/traces"
mkdir -p "temp/logs"
mkdir -p "temp/simulation"

conda run -n dcc --live-stream python3 --version
conda run -n dcc --live-stream python3 kernel_simulation.py --backend="attacc" --workload="attn_single"
conda run -n dcc --live-stream python3 kernel_simulation.py --backend="attacc" --workload="attn_inference"
conda run -n dcc --live-stream python3 kernel_simulation.py --backend="attacc" --workload="gemv_single"
conda run -n dcc --live-stream python3 kernel_simulation.py --backend="attacc" --workload="gemv_inference"
conda run -n dcc --live-stream python3 kernel_simulation.py --backend="attacc" --workload="red_single"
conda run -n dcc --live-stream python3 kernel_simulation.py --backend="hbmpim" --workload="gemv_single"
conda run -n dcc --live-stream python3 kernel_simulation.py --backend="hbmpim" --workload="red_single"
conda run -n dcc --live-stream python3 kernel_simulation.py --backend="hbmpim" --workload="va_single"
conda run -n dcc --live-stream python3 kernel_simulation.py --backend="hbmpim" --workload="relu_single"

conda run -n dcc --live-stream python3 run_kernels.py
conda run -n dcc --live-stream python3 run_inference.py --all

conda run -n dcc --live-stream python3 plot/plot_kernels.py
conda run -n dcc --live-stream python3 plot/plot_inference.py
