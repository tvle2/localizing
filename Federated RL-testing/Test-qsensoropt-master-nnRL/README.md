[![pipeline status](https://gitlab.com/federico.belliardo/qsensoropt/badges/master/pipeline.svg)](https://gitlab.com/federico.belliardo/qsensoropt/-/commits/master)
[![coverage report](https://gitlab.com/federico.belliardo/qsensoropt/badges/master/coverage.svg)](https://gitlab.com/federico.belliardo/qsensoropt/-/commits/master)


# qsensoropt

`qsensoropt` is a library based on Tensorflow 2 for the automatic optimization of adaptive and non-adaptive controls in quantum metrology tasks.

## Docs
Documentation for this project is available on [gitlab pages](https://qsensoropt-federico-belliardo-aafff0229087adae5a915fec60fdc5d37.gitlab.io/).

Along with the description of all the modules,
the documentation contains many examples of use on which the user can base new applications.

## Installation

The following instructions are written for a Linux system.
It is advisable to install `qsensoropt` in a conda environment created for the purpose.

1. If conda is not available on you machine, install it with the following commands
   
   ```
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. Resource the bash configuration with

   ```
   source ~/.bashrc
   ```

3. Create a new conda environment named `qsensoropt`

   ```
   conda create --name qsensoropt python==3.10
   ```

4. Activate the environment just created and upgrade pip

   ```
   conda activate qsensoropt
   pip install --upgrade pip
   ```

5. `qsensoropt` is based on Machine Learning and a such it is best
   used with a GPU. If a GPU is available on your system you can use it by
   installing CUDA and cuDNN

   ```
   conda install -c conda-forge cudatoolkit=11.8
   pip install nvidia-cudnn-cu11==8.6.0.163
   conda install -c nvidia cuda-nvcc
   ```

   Configure the system path with

   ```
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CONDA_PREFIX/bin:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   ```

   followed by 

   ```
   echo 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   ```

   where in the place of `/path/to/cuda/` the complete path to cuda on the machine must be inserted. In most cases, while using conda it is sufficient to specify the directory of the environment `qsensoropt`.
   
   Then restart the environment with

   ```
   conda deactivate
   conda activate qsensoropt
   ```

7. Clone the repository and enter in the `qsensoropt` directory

   ```
   git clone https://gitlab.com/federico.belliardo/qsensoropt.git
   cd qsensoropt
   ```
7. Install the `qsensoropt` library and all its dependencies

   ```
   pip install -e .
   ```

Congratulation! You can now optimize your quantum sensor!
