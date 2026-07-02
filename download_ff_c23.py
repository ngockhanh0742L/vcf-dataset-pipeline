import os

os.environ.setdefault("KAGGLEHUB_CACHE", r"D:\VCF\kagglehub-cache")

import kagglehub


path = kagglehub.dataset_download("xdxd003/ff-c23")
print("Path to dataset files:", path, flush=True)
