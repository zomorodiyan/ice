#!/bin/bash

# Compile the CUDA code
nvcc -arch=compute_75 -code=sm_75 ice.cu -o cu

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the program..."
    # Run the compiled CUDA program
    ./cu
    python plot.py
else
    echo "Compilation failed."
fi

