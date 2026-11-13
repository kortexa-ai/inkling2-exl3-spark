# Public ARM64 CUDA 13 / PyTorch 2.13 base, fixed by content digest.
FROM vllm/vllm-openai@sha256:3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971
WORKDIR /opt/inkling
COPY source-pins.json fetch_sources.py ./
RUN python3 fetch_sources.py && \
    python3 -m pip install --no-deps upstream/build/marisa_trie-*.whl && \
    export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH && \
    export TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=2 && \
    export CPATH=$(python3 -c 'import site; print(site.getsitepackages()[0]+"/nvidia/cu13/include")') && \
    export CPLUS_INCLUDE_PATH=$CPATH C_INCLUDE_PATH=$CPATH && \
    python3 -m pip install --no-deps --no-build-isolation --no-cache-dir upstream/build/exllamav3 && \
    rm -rf upstream/build
COPY math_ops.py kernels.py model.py attention.py runtime.py ./
COPY LICENSE NOTICE ./
COPY tests/test_runtime.py tests/test_runtime.py
COPY download.py launch.py ./
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENTRYPOINT ["python3", "/opt/inkling/runtime.py"]
