#!/bin/bash


check() {
  ver=$($1 --version 2>/dev/null | grep -Po '\d+\.\d+\.\d+' | head -1)
  if [[ -z $ver ]]; then
    echo "ERROR: $1 not installed"
    exit 1
  fi
  if ! printf "%s\n%s\n" "$2" "$ver" | sort -V -C; then
    echo "ERROR: $1 version $ver < required $2"
    exit 1
  fi
  echo "OK: $1 $ver >= $2"
}

check conda 25.6.1
check cmake 3.16.3
check gcc 11.4.0

conda create --name dcc python=3.8 -y
conda run -n dcc --live-stream python3 -m pip install pandas numpy tqdm xgboost==2.1.4 scikit-learn==1.3.2 matplotlib

cd ./src/Ramulator2/
bash ./build.sh
cd ../../
