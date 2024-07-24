# ice 
Step 18 (Speical Topic 2): Phase-field Method
[drzgan.github/CFD/Phase-field Method](https://drzgan.github.io/Python_CFD/Konayashi_1993-main/jax_version/kobayashi_aniso_jax_ZGAN-2.html)

This code is the tutorial's code converted from Python (Jax) to c++ (CUDA).
The tutorial's code reproduce the results presented in *Modeling and numerical simulations of dendritic crystal growth* by Ryo Kobayashi 

linux command to run:

```sh
python ice.py

nvcc ice.cu -o cu
./cu
python plot.py
```


