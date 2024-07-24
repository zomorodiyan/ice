import os
import numpy as np
import matplotlib.pyplot as plt

dt = 0.0001

def read_array(filename, nx, ny):
    data = np.loadtxt(filename)
    return data.reshape((ny, nx))

def plot_data_with_subplots(case_path, nx, ny):
    indices = [400, 1600, 3600]  # Indices for the three subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

    for ax, i in zip(axes, indices):
        filename = os.path.join(case_path, f"output_{i}.txt")
        data = read_array(filename, nx, ny)
        im = ax.imshow(data, cmap='coolwarm', extent=[0, 9, 0, 9], origin='lower', aspect='auto')
        ax.set_title(f"Time: {i * dt:.2f} seconds", fontsize=16)
        ax.set_xlabel('x (distance)', fontsize=14)
        if ax == axes[0]:
            ax.set_ylabel('y (distance)', fontsize=14)
        else:
            ax.set_yticks([])

#    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.08)
#    cbar.set_label('Color bar label', fontsize=14)  # Customize the label as needed
#    cbar.ax.tick_params(labelsize=12)  # Increase color bar tick label size

    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Adjust layout to make space for the color bar
    plt.savefig(os.path.join(case_path, f"{case_name}_subplots.png"))
    plt.close()

case_name = 'ice'
work_path = os.getcwd()
case_path = os.path.join(work_path, case_name)
nx, ny = 300, 300  # Update this to match the nx and ny used in the CUDA code

plot_data_with_subplots(case_path, nx, ny)

