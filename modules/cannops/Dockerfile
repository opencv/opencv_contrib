# User guides
#
# 0. Install Ascend driver on host.
#    (https://www.hiascend.com/en/hardware/firmware-drivers)
#
# 1. Run docker container.
# docker run -it \
#    --name opencv \
#    --device /dev/davinci0 \
#    --device /dev/davinci_manager \
#    --device /dev/devmm_svm \
#    --device /dev/hisi_hdc \
#    -v /usr/local/dcmi:/usr/local/dcmi \
#    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
#    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
#    opencv bash
#
# 2. Check environment.
# npu-smi info
#
# 3. Compile opencv with Ascend NPU backend.
# cmake -DWITH_CANN=1
#
# 4. Run opencv_test_cannops.
# ./bin/opencv_test_cannops

FROM openeuler/openeuler:22.03-lts-sp2

RUN yum install -y \
    git \
    wget \
    gcc \
    g++ \
    cmake \
    make \
    python-pip \
    python3-devel

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy \
    sympy \
    decorator \
    scipy \
    attrs \
    psutil

# Install CANN
RUN wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.RC1/Ascend-cann-toolkit_7.0.RC1_linux-"$(uname -i)".run && \
    chmod +x Ascend-cann-toolkit_7.0.RC1_linux-"$(uname -i)".run && \
    ./Ascend-cann-toolkit_7.0.RC1_linux-"$(uname -i)".run --quiet --install && \
    rm -f ./Ascend-cann-toolkit_7.0.RC1_linux-"$(uname -i)".run

# Install kernel
RUN wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.RC1/Ascend-cann-kernels-310p_7.0.RC1_linux.run && \
    chmod +x Ascend-cann-kernels-310p_7.0.RC1_linux.run && \
    ./Ascend-cann-kernels-310p_7.0.RC1_linux.run --quiet --install && \
    rm -f ./Ascend-cann-kernels-310p_7.0.RC1_linux.run

ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH:/usr/lib64
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:$LD_LIBRARY_PATH
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:$PATH
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
