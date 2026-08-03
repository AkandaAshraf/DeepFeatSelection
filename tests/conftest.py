import os

# Keep TensorFlow quiet and deterministic-ish during tests. Must be set before
# the first tensorflow import, which pytest triggers on collection.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: trains a model; takes tens of seconds")
