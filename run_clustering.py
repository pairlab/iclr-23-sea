import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pathlib

data_path = pathlib.Path("~/logs").expanduser() / 'polybeast' / "crafter" / "run" / "only_one"