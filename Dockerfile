# The GPU image already carries a matching CUDA runtime and cuDNN, so the
# plain `tensorflow` wheel is installed rather than the `[and-cuda]` extra.
# For a CPU-only build, swap the tag for `tensorflow/tensorflow:2.20.0`.
FROM tensorflow/tensorflow:2.20.0-gpu

WORKDIR /app

COPY pyproject.toml README.md ./
COPY deepfeatselect ./deepfeatselect
RUN pip install --no-cache-dir .

COPY Data ./Data
COPY scripts ./scripts

ENTRYPOINT ["deepfeatselect"]
CMD ["--n-models", "20"]
