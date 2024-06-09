# [Preprocessing the data]

# [Libraries]{Preprocessing}
import numpy as np

# [Library]{Randomness}
import random

# [Library]{Signal processing}
from scipy import signal as sig
from sklearn.preprocessing import MinMaxScaler

# [Library]{Homology}
import gtda.homology as hl
import gtda.time_series as ts

# [Library]{Save and load data}
import os
from pathlib import Path

# [Library]{Gravitational waves}
from Dataset.generate_datasets import make_gravitational_waves

# [Class]{GWDataset}
class GWDataset:
    # [Constructor]{Preprocessing}
    def __init__(
            self,
            PATH : str = './Dataset/',
            n_signals : int = 1000,
            R : list = (0.075,0.65),
            xi : float = 1
        ):
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        self.PATH = PATH
        self.n_signals = n_signals
        self.R = R
        (self.noisy_signals, self.gw_signals, self.labels) = make_gravitational_waves(
           path_to_data=Path(PATH), n_signals=n_signals, r_min=R[0], r_max=R[1], n_snr_values=1
        )
        # Apply the preprocessing of min-max scaler
        self.scaler = MinMaxScaler()
        self.noisy_signals = self.scaler.fit_transform(self.noisy_signals)
        self.gw_signals = self.scaler.fit_transform(self.gw_signals)
        self.labels = self.labels.reshape(-1,1)
        pass

    # [Method]{get_data}
    def get_data(self):
        return self.noisy_signals, self.gw_signals, self.labels
    
    # [Method]{getTrainData}
    def getTrainData(self):
        return self.noisy_signals, self.labels