wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
chmod +x cuda_12.8.1_570.124.06_linux.run
./cuda_12.8.1_570.124.06_linux.run --silent --toolkit --installpath=$HOME/cuda --override
export PATH=$HOME/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version
nvidia-smi
