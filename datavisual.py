import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

fig_x = 8
fig_y = 8

def csv_to_lists(path):

    df = pd.read_csv(path, header=None)

    arr = df.to_numpy()

    episodes = []

    percent_completed = []

    game_status = []

    time_remaining = []

    for i in range(arr.shape[0]):
        episodes.append(i + 1)
        percent_completed.append(arr[i, 0])
        game_status.append(arr[i, 1])
        time_remaining.append(arr[i, 2])

    return episodes, percent_completed, game_status, time_remaining

ep, pc, gs, tr = csv_to_lists("data/doubleqlearning_run4.csv")

X_Y_Spline = make_interp_spline(ep, pc)

X_ = np.linspace(min(ep), max(ep), 250)
Y_ = X_Y_Spline(X_)

plt.figure(figsize=(fig_x, fig_y))


plt.plot(ep, pc, color="darkgreen")
plt.plot(X_, Y_, color="yellow")

plt.show()