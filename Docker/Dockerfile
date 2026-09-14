FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV PYTHONPATH=/app
ENV AITEM_PROJECTS_DIR=/data/projects
ENV AITEM_DATASETS_DIR=/data/datasets
ENV AITEM_WEIGHTS_DIR=/data/weights
ENV PATH=/opt/conda/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        g++ \
        ninja-build \
        gosu \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
# timm: Detectron2 ViTDet. ultralytics: YOLO-seg engine (not a Detectron2 extra).
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir opencv-python-headless \
    && pip install --no-cache-dir "git+https://github.com/facebookresearch/detectron2.git"

COPY app /app/app
COPY engine /app/engine
COPY core /app/core
COPY tests /app/tests
COPY pytest.ini /app/pytest.ini
COPY Docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
