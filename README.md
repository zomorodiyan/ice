# ice 
Step 18 (Speical Topic 2): Phase-field Method
[drzgan.github/CFD/Phase-field Method](https://drzgan.github.io/Python_CFD/Konayashi_1993-main/jax_version/kobayashi_aniso_jax_ZGAN-2.html)

This code is the tutorial's code converted from Python (Jax) to c++ (CUDA).
The tutorial's code reproduce the results presented in *Modeling and numerical simulations of dendritic crystal growth* by Ryo Kobayashi 

run run.sh in terminal:
```sh
nvcc -arch=compute_75 -code=sm_75 ice.cu -o cu
./cu
python plot.py
python plot_trio.py
```

## Figure 7

### delta = 0.000
![Figure 7: delta = 0.000](figures/fig7_delta_0.000.png)

### delta = 0.005
![Figure 7: delta = 0.005](figures/fig7_delta_0.005.png)

### delta = 0.010
![Figure 7: delta = 0.010](figures/fig7_delta_0.010.png)

### delta = 0.020
![Figure 7: delta = 0.020](figures/fig7_delta_0.020.png)

### delta = 0.050
![Figure 7: delta = 0.050](figures/fig7_delta_0.050.png)


## Figure 8

### K = 0.8
![Figure 8: K = 0.8](figures/fig8_K_0.8.png)

### K = 1.0
![Figure 8: K = 1.0](figures/fig8_K_1.0.png)

### K = 1.2
![Figure 8: K = 1.2](figures/fig8_K_1.2.png)

### K = 1.4
![Figure 8: K = 1.4](figures/fig8_K_1.4.png)

### K = 1.6
![Figure 8: K = 1.6](figures/fig8_K_1.6.png)

### K = 2.0
![Figure 8: K = 2.0](figures/fig8_K_2.0.png)


