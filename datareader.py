import os
import numpy as np
import pandas as pd
from tqdm import tqdm

class DataReader:
    def __init__(
            self, format="numpy", sampling_rate=100, **kwargs
    ):
        self.buffer = {}

        if format in ["numpy"]:
            self.data_dir = kwargs["data_dir"]
            try:
                csv = pd.read_csv(kwargs["data_list"], header=0,sep="[,|\s+]", engine="python")
            except:
                csv = pd.read_csv(kwargs["data_list"], header=0, sep="\t")
            self.data_list = csv["fname"]
            self.num_data = len(self.data_list)

    def __len__(self):
        return self.num_data
    
    def read_numpy(self,fname):
        # Do you need this? It looks mostly like stuff to get metadata specifically for PN?
