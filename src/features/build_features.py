import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
%matplotlib inline
import matplotlib.pyplot as plt
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
df_temporal=df_squared.copy()
numabs=NumericalAbstraction()
predictor_columns=predictor_columns+["acc_r","gyr_r"]
df_temporal_list=[]
for s in df_temporal["set"].unique():
    subset=df_temporal[df_temporal["set"]==s].copy()
    for col in predictor_columns:
        subset=numabs.abstract_numerical(subset,[col],window_size=5,aggregation_function="mean")
        subset=numabs.abstract_numerical(subset,[col],window_size=5,aggregation_function="std")
        df_temporal_list.append(subset)   
df_temporal=pd.concat(df_temporal_list)        


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
# --------------------------------------------------------------
# Frequency Features
# --------------------------------------------------------------

# Reset the index because the Fourier Transform functions expect a discrete index
df_freq = df_temporal.copy().reset_index()

# Initialize the Fourier Transformation feature abstraction class
freq_apps = FourierTransformation()

# Define the sampling rate and window size parameters
fs = int(1000 / 200)   # Sampling rate: 5 samples per second
ws = int(2800 / 200)   # Window size: 14 steps (approx. 2.8 second repetition)

# Process the frequency features set by set to prevent data leakage between sessions
df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier Transformations to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = freq_apps.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

# Concatenate the processed subsets back into a single DataFrame
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)")

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# Drop any missing values introduced by the rolling window boundaries
df_freq = df_freq.dropna()

# Reduce dataset correlation/overfitting by eliminating 50% of row overlap
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

# Define the specific columns to run the clustering algorithm on
cluster_columns = ["acc_x", "acc_y", "acc_z"]
df_cluster = df_freq.copy()

# Code segment used during the Elbow Method optimization loop
k_values = range(2, 10)
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    subset = df_cluster[cluster_columns]
    kmeans.fit(subset)
    inertias.append(kmeans.inertia_)

# Train the final K-Means model using the optimized 5 clusters found from the elbow plot
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plot clusters colored by K-Means unsupervised groups
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# Plot accelerometer data split by actual exercise labels for validation comparison
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")




# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering


# 1. Prepare data

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------