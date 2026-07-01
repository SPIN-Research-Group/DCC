<p align="center">
  <img src="logo.png" alt="DCC logo" width="300">
</p>

<h2 align="center">
  DCC: Data-Centric Compilation of Machine Learning Kernels <br>  
  for Processing-In-Memory Architectures 
</h2>

[<i>DCC</i>]([https://arxiv.org/pdf/2511.20834](https://arxiv.org/pdf/2511.15503)) is the first data-centric Machine Learning compiler for Processing-In-Memory (PIM) architectures.

High-performance Host processors (e.g., GPUs) can integrate Processing-In-Memory (PIM) devices, which can accelerate memory-intensive kernels of Machine Learning (ML) models, including Large Language Models (LLMs), by leveraging the large memory bandwidth available at PIM cores. However, Host processor and PIM cores require different data layouts: Host processor needs consecutive elements distributed across DRAM banks, while PIM cores need consecutive elements within their local banks. This necessitates data rearrangements in ML kernel execution that pose significant performance and programmability challenges, further exacerbated by the need to support diverse PIM devices (e.g., Samsung HBM-PIM, SK Hynix GDDR6-AiM). Current compilation approaches lack systematic optimization for diverse ML kernels and multiple PIM devices, and may largely ignore data rearrangement costs during the compute code optimization step. We demonstrate that data rearrangements and compute code optimization are interdependent, and need to be jointly optimized during the tuning process. 

DCC is the first data-centric ML compiler for PIM systems that jointly co-optimizes data rearrangements and compute code in a unified tuning process to enable high performance execution. DCC integrates a multi-layer PIM abstraction that enables various data distribution strategies on different PIM backends. DCC enables effective co-optimization of data partitioning strategies with compute loop partitioning schemes. DCC applies PIM-specific code optimizations, and leverages a fast and accurate performance prediction model to select the bestperforming code schedule for a given kernel on a target PIM architecture. DCC provides significant performance benefits across various ML kernels, LLM models, and PIM backends.

## Cite DCC

Please use the following citations to cite DCC, if you find this repository useful:

Bibtex entries for citation:
```
@inproceedings{Yang2026DCC,
author = {Yang, Peiming and Durvasula, Sankeerth and Fernandez, Ivan and Sadrosadati, Mohammad and Mutlu, Onur and Pekhimenko, Gennady and Giannoula, Christina},
title = {DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures},
year = {2026},
booktitle = {Proceedings of the 53rd Annual International Symposium on Computer Architecture},
location = {Raleigh, NC, USA},
series = {ISCA '26}
}
```


## Hardware Requirements
The artifact has been rigorously tested and validated on server-class hardware meeting the following minimum specifications:

* CPU: x86-64 architecture with a minimum of 64 hardware threads (32 physical cores with simultaneous multithreading enabled), 128GB of main system memory, and at least 128GB of available disk storage (SSD recommended for optimal I/O performance during simulation)
  
* GPU: NVIDIA discrete GPU with a minimum compute capability (SM version) of 8.0 (Ampere architecture or newer, e.g., A100, RTX 30xx/40xx, A40) and a minimum of 8GB of dedicated GPU memory (VRAM)


## Software Requirements
Before running the scripts, ensure the following dependencies are installed:
- conda >= 25.6.1 (Anaconda or Miniconda distribution)
- CMake == 3.16.3 (**strict requirement**)
- GCC == 11.4.0 (**strict requirement**)
- CUDA 12.x
  
## Execution Steps
#### **Step 1: Prepare environment**  
   First, complete the following pre-configuration steps (steps 1 and 2 can be skipped if the required software is already installed on your system):
   1. Install Anaconda 25.6.1 by following the official instructions at: https://www.anaconda.com/docs/getting-started/main. You may need to run `conda tos accept` after installation to enable conda to create new virtual environment, please refer to https://www.anaconda.com/docs/getting-started/tos-plugin.
   2. Download and install CMake 3.16.3 by following the official instructions at: https://cmake.org/download/. Please note that we have not extensively tested various Cmake versions. If the building fails due to CMake, we **recommend** using and building our artifact with CMake 3.16.3.
   3. Download and unzip the source code of DCC_Artifact to your local machine.
   4. Then, run the following command in the **code root directory** to automatically configure the isolated Anaconda environment, install all required Python packages, and build the DCC compiler backend:
      ```bash
      bash setup.sh
      ```
      This script will handle all dependency resolution, environment activation, and compilation steps. Ensure you have a stable internet connection during this process.

#### **Step 2: Run all experiments**:
   After the environment setup completes successfully, execute the main experiment pipeline with:
   ```bash
   bash run_experiments.sh
   ```
   > **Note**: The full end-to-end experiment suite, including all kernel simulations, compilation passes, and result generation, is expected to take 7–10 days to complete on the recommended hardware. The script is fully interruptible: if execution is paused, terminated, or fails at any point, you may simply re-launch the script run_experiments.sh at any time. The script will automatically detect all unfinished simulations and/or compilation tasks, resume from the last completed checkpoint, and re-run only the incomplete steps and simulations, eliminating the need to restart the entire pipeline from scratch.

#### **Step 3: View outputs**:
   All experiment results and generated figures are saved in the following directories:
   * Raw results are saved in `./results`
   * Final plots (reproducing Figures 6, 7, 9 of the paper) are saved in `./figures` directory.
