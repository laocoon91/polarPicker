import os
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

class DataReader:
    def __init__(self, format="numpy", sampling_rate=100, **kwargs):
        self.buffer = {}
        self.format = format

        if format == "numpy":
            self.data_dir = kwargs["data_dir"]
            try:
                csv = pd.read_csv(kwargs["data_list"], header=0,sep="[,|\s+]", engine="python")
            except:
                csv = pd.read_csv(kwargs["data_list"], header=0, sep="\t")
            self.data_list = csv["fname"]
            self.num_data = len(self.data_list)
        elif format == "hdf5":
            self.h5 = h5py.File(kwargs["hdf5_file"], "r", libver="latest")
            self.h5_data
        else:
            raise (f"{format} not supported")
    
    def read_numpy(self,fname):
    
        if fname not in self.buffer:
            npz = np.load(fname)
            meta = {}
            meta["data"] = npz["data"][:,2] # Z-comp
            meta["p_idx"] = npz["p_idx"]
        else:
            meta = self.buffer[fname]

        return meta
    
class DataReader_train(DataReader): 
    def __init__(self, format="numpy",**kwargs):
        super().__init__(format=format,**kwargs)

    def get_numpy_data(self):
    #def __getitem__(self):

        base_name = self.data_list

        Xarr = []
        yarr = []

        pbar = tqdm(total=len(base_name))

        for rec in base_name:
            pbar.update()
            if self.format == "numpy":
                meta = self.read_numpy(os.path.join(self.data_dir,rec))
            data = meta["data"]
            p_idx = meta["p_idx"]
            Xarr.append(data[p_idx-32:p_idx+32])
            #yarr.append() # need to read in polarity values somewhere 

        #return (np.array(Xarr), to_categorical((np.array(yarr) == 'positive').astype(int)))
        return (np.array(Xarr))
