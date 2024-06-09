# [Preprocessing the data]

# [Libraries]{Preprocessing}
import numpy as np

# [Library]{Randomness}
import random

# [Library]{Signal processing}
from scipy import signal as sig

# [Library]{Homology}
from gtda.diagrams import Scaler
from gtda.metaestimators import CollectionTransformer
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import TakensEmbedding
from gtda.pipeline import Pipeline

# [Library]{File managment}
import os

# [Library]{Dimensionality reduction}
from sklearn.decomposition import PCA
# [Library]{MinMaxScaler}
from sklearn.preprocessing import MinMaxScaler

# [Class]{Preprocessing}
class GWPreprocessing:
    # [Constructor]{Preprocessing}
    def __init__(
            self,
            dimension : int = 10,
            delay : int = 10,
            stride : int = 1,
            components : int = 1,
        ):
        # [Parameters]{Preprocessing}
        self.embedding = dimension
        self.delay = delay
        self.stride = stride
        self.components = components

        # [Methods]{Change data}
        self.SEmbedding = TakensEmbedding(
            
            time_delay=delay,
            dimension=dimension,
            stride=stride
        )
        # [Methods]{Dimensionality reduction}
        self.PCA = PCA(n_components=components)
        self.PCA = CollectionTransformer(self.PCA, n_jobs=-1)

        # [Methods]{Pipeline}
        self.DReductionPipeline = Pipeline([
            ("embedding", self.SEmbedding),
            ("pca", self.PCA),
            #("scaler", self.Scaler)
        ])

        # [Methods]{Homology}
        self.VRips = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=[0, 1],
            n_jobs=-1
        )
        
    def fit(self, X, y=None):
        # [Methods]{Fit}
        Data = self.DReductionPipeline.fit_transform(X)
        Homology = self.VRips.fit_transform(Data)
        return Data, Homology
    
    