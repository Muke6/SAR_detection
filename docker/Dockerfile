ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"


FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"



RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN  apt-get clean
RUN apt-get update && apt-get install -y --no-install-recommends \
		 libglib2.0-0 libsm6 libxrender-dev libxext6 \
         build-essential \
         cmake \
         curl \
         libgl1-mesa-glx \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*
# 将程序复制容器内，表示在/workspace路径下
# mmcv
RUN pip install opencv-python -i https://pypi.mirrors.ustc.edu.cn/simple/
RUN pip install mmcv-full==1.3.9 -i https://pypi.douban.com/simple/

COPY .  /workspace
# 确定容器启动时程序运行路径
WORKDIR /workspace

# pip3 = pip
RUN pip install cython -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip --default-timeout=100 install -r requirements/build.txt -i https://pypi.douban.com/simple/
RUN pip install pandas matplotlib -i https://pypi.mirrors.ustc.edu.cn/simple
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -e .
# 确定容器启动命令。这里列举python示例，python表示编译器，xxx.py表示执行文件，input_path和output_path为容器内绝对路径，测评时会自动将测试数据挂载到容器内/input_path路径，不需要修改；
CMD ["python", "run.py", "/input_path", "/output_path"]



