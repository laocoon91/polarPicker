import numpy as np
import pandas as pd
from predict import polarPredict

X = np.load("./data/Portland_data_polarityInput_newdownload.npy")
polarpred = polarPredict(mode="predict",model="polarityModel_20240803.keras")

predictions =polarpred.predict(X)

olist = pd.DataFrame(predictions)
olist.to_csv('Portland_data_polarityInput_predictions_undecided_newdownload.csv',index=False,header=False)
