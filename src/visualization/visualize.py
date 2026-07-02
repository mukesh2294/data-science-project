import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
labels=df["label"].unique()
participants=df["participant"].unique()
for label in labels:
    for participant in participants:
        combined_plot_df= df.query(f"label=='{label}'")\
                            .query(f"participant=='{participant}'")\
                            .reset_index()     
        if len(combined_plot_df)>0:
            fig, ax = plt.subplots(nrows=2,sharex=True, figsize=(20, 10))     
            combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])
            plt.title(f"Participant: {participant} | Exercise: {label}")
            ax[1].set_xlabel("samples")  
            ax[0].legend(loc="upper center", ncol=3,shadow=True, fancybox=True)                     
            ax[1].legend(loc="upper center", ncol=3,shadow=True, fancybox=True)
            plt.savefig(f"../../reports/figures/{label.title()}({participant}).png")