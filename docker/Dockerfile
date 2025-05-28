FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

# set timeout
RUN /bin/echo -e "Acquire::http::Timeout \"300\";\n\
Acquire::ftp::Timeout \"300\";" >> /etc/apt/apt.conf.d/99timeout

# update and install necessary tools
RUN apt update && apt upgrade -y \
   && apt install -y \
          build-essential libbz2-dev libdb-dev \
          libreadline-dev libffi-dev libgdbm-dev liblzma-dev \
          libncursesw5-dev libsqlite3-dev libssl-dev \
          zlib1g-dev uuid-dev tk-dev liblapack-dev \
          graphviz fonts-humor-sans git libjpeg-dev \
          nvidia-cuda-toolkit swig libgflags-dev wget

# remove cache
RUN apt clean -y \
  && apt autoremove -y \
  && apt autoclean -y \
  && rm /var/lib/apt/lists/* -rf


# install cmake
ARG CMAKE_VER=4.0.2
RUN wget "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}.tar.gz" \
  && tar -zxvf cmake-${CMAKE_VER}.tar.gz \
  && cd cmake-${CMAKE_VER} \
  && ./bootstrap \
  && make \
  && make install


# install python
ARG VER=3.12.10
RUN wget "https://www.python.org/ftp/python/${VER}/Python-${VER}.tgz" \
    && tar -xzvf Python-${VER}.tgz \
    && cd Python-${VER} \
    && ./configure --enable-optimizations --with-lto \
    && make \
    && make install

# install pip
RUN wget "https://bootstrap.pypa.io/get-pip.py" \
    && python3 get-pip.py

RUN pip install "numpy<2" setuptools

# install faiss
# CMAKE_CUDA_ARCHITECTURES (Compute capability): V100=70, A100=80, H100=90
# https://developer.nvidia.com/cuda-gpus

RUN git clone https://github.com/facebookresearch/faiss \
   && cd faiss \
   && cmake -B build . \
        -DFAISS_ENABLE_GPU=ON \
        -DFAISS_ENABLE_PYTHON=ON \
        -DFAISS_ENABLE_RAFT=OFF \
        -DBUILD_TESTING=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DFAISS_ENABLE_C_API=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DFAISS_OPT_LEVEL=avx2 \
        -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 \
        -DCMAKE_CUDA_ARCHITECTURES="80" \
        -DPython3_EXECUTABLE=/usr/local/bin/python3.12 \
        -DPython3_LIBRARY=/usr/local/lib \
        -DPython3_INCLUDE_DIR=/usr/local/include/python3.12 \
        -DCMAKE_CUDA_COMPILER=$(which nvcc) \
   && make -C build -j faiss \
   && make -C build -j swigfaiss \
   && cd build/faiss/python \
   && python3 setup.py install


RUN pip install -U pip wheel
RUN pip install -U jupyter
RUN pip install numpy matplotlib seaborn umap-learn optuna pandas scipy \
                scikit-learn opencv-python POT
RUN pip install Cython h5py pyzmq tornado pillow-simd six \
                easydict pyyaml requests multiprocess
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install pytorch-ignite timm einops
RUN pip install wilds

RUN pip install -U "numpy<2"
