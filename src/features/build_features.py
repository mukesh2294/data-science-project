import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df=pd.read_pickle("../../data/interim/data_outliers_removed_chauvenet.pkl")

predictor_columns=list(df.columns[:6])
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] = 100
plt.style.use("fivethirtyeight")
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# -----------------------------------------
# ---------------------
for col in predictor_columns:
    df[col]=pd.to_numeric(df[col], errors="coerce")
    df[col]=df[col].interpolate()
df.info()
df[df["set"]==25]["acc_y"].plot()
df[df["set"]==50]["acc_y"].plot()
duration=df[df["set"]==1].index[-1]-df[df["set"]==1].index[0]
duration.seconds
for s in df["set"].unique():
    start=df[df["set"]==s].index[0]
    stop=df[df["set"]==s].index[-1]
    duration=stop-start
    df.loc[df["set"]==s, "duration"] = duration
duration_df=df.groupby(["category"])["duration"].mean()
duration_df.iloc[0]/5
duration_df.iloc[1]/10
# --------------------------------------------------------------
df_lowpass=df.copy()
lowpass=LowPassFilter()
fs=1000/200
cutoff=1.3
print(df_lowpass["acc_x"].dtype)
df_lowpass=lowpass.low_pass_filter(df_lowpass,"acc_x",fs,cutoff,order=5)
for col in predictor_columns:
    df_lowpass=lowpass.low_pass_filter(df_lowpass,col,fs,cutoff,order=5)
    df_lowpass[col]=df_lowpass[col+"_lowpass"]
    del df_lowpass[col+"_lowpass"]
  print(df_lowpass) 
    
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca=df_lowpass.copy()
pca=PrincipalComponentAnalysis()
pc_values=pca.determine_pc_explained_variance(df_pca,predictor_columns)
plt.figure(figsize=(10,5))
plt.plot(range(1,len(pc_values)+1),pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show
df_pca=pca.apply_pca(df_pca,predictor_columns,number_comp=3)
print(df_pca)



# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared=df_pca.copy()
acc_r=np.sqrt(df_squared["acc_x"]**2+df_squared["acc_y"]**2+df_squared["acc_z"]**2)
gyr_r=np.sqrt(df_squared["gyr_x"]**2+df_squared["gyr_y"]**2+df_squared["gyr_z"]**2)
df_squared["acc_r"]=acc_r
df_squared["gyr_r"]=gyr_r
df_squared[["acc_r","gyr_r"]].where(df_squared["set"]==35).plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------