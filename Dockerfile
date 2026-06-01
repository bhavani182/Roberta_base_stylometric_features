FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY single_fusion.py /app/single_fusion.py
COPY obfuscated_data/ /app/obfuscated_data/
COPY Roberta_base_with_stylometric_features.ipynb /app/Roberta_base_with_stylometric_features.ipynb

CMD ["python", "/app/single_fusion.py"]