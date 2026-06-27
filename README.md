# DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory


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
