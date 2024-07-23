import os
import numpy as np
import matplotlib.pyplot as plt
dt=0.0001

def read_array(filename, nx, ny):
    data = np.loadtxt(filename)
    return data.reshape((ny, nx))
def plot_data(case_path, nx, ny):
    for i in range(200, nIter + 1, 200):
        filename = os.path.join(case_path, f"ice/output_{i}.txt")
        data = read_array(filename, nx, ny)
        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap='viridis')
        plt.colorbar()
        plt.savefig(os.path.join(case_path, f"{case_name}_{i * dt:.2f}.png"))
        plt.close()

case_name = 'ice'
work_path = os.getcwd()
case_path = os.path.join(work_path, case_name)
nx, ny = 300, 300  # Update this to match the nx and ny used in the CUDA code
nIter = int(0.36 / 0.0001)

plot_data(case_path, nx, ny)
