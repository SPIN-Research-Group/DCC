#!/bin/bash

# Exit on error
set -e


cd ./AttAcc
# Clean up previous build
if [ -d "build" ]; then
  rm -rf build
fi

# Build
mkdir -p build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.16
make -j
cp ./ramulator2 ../../ramulator2-AttAcc
cd ../../

cd ./HBMPIM
# Clean up previous build
if [ -d "build" ]; then
  rm -rf build
fi

# Build
mkdir -p build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.16
make -j
cp ./ramulator2 ../../ramulator2-HBMPIM
cd ../../


