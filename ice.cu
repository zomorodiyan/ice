#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

#define J 6
#define K 2.0
#define T_eq 1.0
#define a 0.01
#define alpha 0.9
#define delta 0.04
#define dt 0.0001
#define eps_bar 0.01
#define gamma 10.0
#define t_OFF 0.50
#define tau 0.0003
#define nIter int(t_OFF/dt)+1

const int nx = 300; 
const int ny = 300;
const float hx = 0.03f;
const float hy = 0.03f;

// Error checking macro
#define CUDA_CHECK_ERROR() {                                           \
    cudaError_t err = cudaGetLastError();                              \
    if (err != cudaSuccess) {                                          \
        printf("CUDA Error: %s\n", cudaGetErrorString(err));           \
        exit(-1);                                                      \
    }                                                                  \
}

__global__ void init_curand(curandState* state, unsigned long seed, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * nx + idx;
    if (idx < nx && idy < ny) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void grad(float* m, float* f_x, float* f_y, float dx, float dy, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < nx && idy < ny) {
        int id = idy * nx + idx;
        int left = idy * nx + ((idx - 1 + nx) % nx);
        int right = idy * nx + ((idx + 1) % nx);
        int top = ((idy - 1 + ny) % ny) * nx + idx;
        int bottom = ((idy + 1) % ny) * nx + idx;
        f_x[id] = (m[right] - m[left]) / (2.0 * dx);
        f_y[id] = (m[bottom] - m[top]) / (2.0 * dy);
    }
}

__global__ void laplace(float* m, float* result, float hx, float hy, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < nx && idy < ny) {
        int id = idy * nx + idx;

        int left = idy * nx + ((idx - 1 + nx) % nx);
        int right = idy * nx + ((idx + 1) % nx);
        int top = ((idy - 1 + ny) % ny) * nx + idx;
        int bottom = ((idy + 1) % ny) * nx + idx;

        result[id] = (m[top] + m[bottom] - 2.0 * m[id]) / (hx * hx) + 
                     (m[left] + m[right] - 2.0 * m[id]) / (hy * hy);
    }
}

__global__ void get_theta(float* f_x, float* f_y, float* theta, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < nx && idy < ny) {
        int id = idy * nx + idx;

        theta[id] = 0.0;
        if (f_x[id] == 0 && f_y[id] > 0)
            theta[id] = 0.5 * M_PI;
        else if (f_x[id] == 0 && f_y[id] < 0)
            theta[id] = 1.5 * M_PI;
        else if (f_x[id] > 0 && f_y[id] < 0)
            theta[id] = 2 * M_PI + atan(f_y[id] / f_x[id]);
        else if (f_x[id] > 0 && f_y[id] > 0)
            theta[id] = atan(f_y[id] / f_x[id]);
        else if (f_x[id] < 0)
            theta[id] = M_PI + atan(f_y[id] / f_x[id]);
    }
}

__global__ void get_eps(float* theta, float* eps, float* eps_prime, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nx && idy < ny) {
        int id = idy * nx + idx;

        eps[id] = eps_bar * (1 + delta * cos(J * theta[id]));
        eps_prime[id] = -eps_bar * J * delta * sin(J * theta[id]);
    }
}

__global__ void phase_field(curandState* state, float* eps, float* eps_prime, float* eps2_x, float* eps2_y, float* p, float* p_x, float* p_y, float* p_lap, float* p_dif, float* T, float dx, float dy, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * nx + idx;
    if (idx < nx && idy < ny) {

        int left = idy * nx + ((idx - 1 + nx) % nx);
        int right = idy * nx + ((idx + 1) % nx);
        int top = ((idy - 1 + ny) % ny) * nx + idx;
        int bottom = ((idy + 1) % ny) * nx + idx;

        float part1 = (eps[right] * eps_prime[right] * p_y[right] - eps[left] * eps_prime[left] * p_y[left]) / (2.0 * dx);
        float part2 = (eps[bottom] * eps_prime[bottom] * p_x[bottom] - eps[top] * eps_prime[top] * p_x[top]) / (2.0 * dy);
        float part3 = eps2_x[id] * p_x[id] + eps2_y[id] * p_y[id];
        float part4 = eps[id] * eps[id] * p_lap[id];

        float m = alpha / M_PI * atan(gamma * (T_eq - T[id]));

        float term1 = -part1 + part2 + part3 + part4;
        float term2 = p[id] * (1 - p[id]) * (p[id] - 0.5 + m);

        // Add noise using curand
        curandState localState = state[id];
        float noise = a * p[id] * (1 - p[id]) * (curand_uniform(&localState) - 0.5);
        
        p_dif[id] = dt / tau * (term1 + term2 + noise);

        p[id] = p[id] + p_dif[id];
    }
}

__global__ void T_field(float* T, float* p_dif, float* T_lap, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nx && idy < ny) {
        int id = idy * nx + idx;
        T[id] = T[id] + dt*T_lap[id] + K*p_dif[id];
    }
}

__global__ void zero_flux_BC(float* arr, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < nx && idy < ny) {
        int id = idy * nx + idx;

        if (idx == 0)
            arr[id] = arr[idy * nx + 1];
        else if (idx == nx - 1)
            arr[id] = arr[idy * nx + (nx - 2)];
        if (idy == 0)
            arr[id] = arr[1 * nx + idx];
        else if (idy == ny - 1)
            arr[id] = arr[(ny - 2) * nx + idx];
    }
}

__global__ void elementwise_multiply(float* aa, float* bb, float* result, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * nx + idx;

    if (idx < nx && idy < ny) {
        result[id] = aa[id] * bb[id];
    }
}

void printArray(const char* name, float* array, int nx, int ny) {
    std::cout << name << ":\n";
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            std::cout << array[i * nx + j] << " ";
        }
        std::cout << "\n";
    }
}

void saveArray(const char* filename, float* array, int nx, int ny) {
    std::ofstream file(filename);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            file << array[i * nx + j] << " ";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    float t = 0.0;

    // Initialize arrays
    float *T, *p, *theta, *p_x, *p_y, *eps, *eps_prime, *eps2_x, *eps2_y;
    float *d_T, *d_T_lap, *d_p, *d_theta, *d_p_x, *d_p_y, *d_p_dif, *d_p_lap, *d_eps, *d_eps2, *d_eps_prime, *d_eps2_x, *d_eps2_y;
    curandState* d_state;

    T = (float*)calloc(nx * ny, sizeof(float));
    p = (float*)calloc(nx * ny, sizeof(float));
    theta = (float*)calloc(nx * ny, sizeof(float));
    p_x = (float*)calloc(nx * ny, sizeof(float));
    p_y = (float*)calloc(nx * ny, sizeof(float));
    eps = (float*)calloc(nx * ny, sizeof(float));
    eps_prime = (float*)calloc(nx * ny, sizeof(float));
    eps2_x = (float*)calloc(nx * ny, sizeof(float));
    eps2_y = (float*)calloc(nx * ny, sizeof(float));

    // Define the center of the grid
    int centerX = nx / 2;
    int centerY = ny / 2;
    int radius = 5;

    // Loop over each element in the grid
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            // Check if the current element is within the nucleation region
            if ((i - centerX) * (i - centerX) + (j - centerY) * (j - centerY) < radius * radius) {
                // Set the value to 1 for nucleation
                p[j * nx + i] = 1.0;
            }
        }
    }

    cudaMalloc((void**)&d_T, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_T_lap, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_p, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_p_dif, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_p_lap, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_theta, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_p_x, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_p_y, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_eps, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_eps2, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_eps_prime, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_eps2_x, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_eps2_y, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_state, nx * ny * sizeof(curandState));

    cudaMemcpy(d_T, T, nx * ny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, nx * ny * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    init_curand<<<numBlocks, threadsPerBlock>>>(d_state, time(0), nx, ny);
    CUDA_CHECK_ERROR();
    cudaDeviceSynchronize();

    for (int i = 0; i <= nIter; ++i) {
        grad<<<numBlocks, threadsPerBlock>>>(d_p, d_p_x, d_p_y, hx, hy, nx, ny);
        cudaDeviceSynchronize();

        laplace<<<numBlocks, threadsPerBlock>>>(d_p, d_p_lap, hx, hy, nx, ny);
        cudaDeviceSynchronize();

        get_theta<<<numBlocks, threadsPerBlock>>>(d_p_x, d_p_y, d_theta, nx, ny);
        cudaDeviceSynchronize();

        get_eps<<<numBlocks, threadsPerBlock>>>(d_theta, d_eps, d_eps_prime, nx, ny);
        cudaDeviceSynchronize();

        elementwise_multiply<<<numBlocks, threadsPerBlock>>>(d_eps, d_eps, d_eps2, nx, ny);
        cudaDeviceSynchronize();

        grad<<<numBlocks, threadsPerBlock>>>(d_eps2, d_eps2_x, d_eps2_y, hx, hy, nx, ny);
        cudaDeviceSynchronize();

        phase_field<<<numBlocks, threadsPerBlock>>>(d_state, d_eps, d_eps_prime, d_eps2_x, d_eps2_y, d_p, d_p_x, d_p_y, d_p_lap, d_p_dif, d_T, hx, hy, nx, ny);
        cudaDeviceSynchronize();

        zero_flux_BC<<<numBlocks, threadsPerBlock>>>(d_p, nx, ny);
        cudaDeviceSynchronize();

        laplace<<<numBlocks, threadsPerBlock>>>(d_T, d_T_lap, hx, hy, nx, ny);
        cudaDeviceSynchronize();

        T_field<<<numBlocks, threadsPerBlock>>>(d_T, d_p_dif, d_T_lap, nx, ny);
        cudaDeviceSynchronize();

        zero_flux_BC<<<numBlocks, threadsPerBlock>>>(d_T, nx, ny);
        cudaDeviceSynchronize();

        if (i % 800 == 0) {
            std::cout << "step/"<<nIter<<": " << i << "\n";
            cudaMemcpy(p, d_p, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

            char filename[100];
            sprintf(filename, "ice/output_%d.txt", i);
            saveArray(filename, p, nx, ny);
        }

        t += dt;
    }

    cudaFree(d_T);
    cudaFree(d_p);
    cudaFree(d_theta);
    cudaFree(d_p_x);
    cudaFree(d_p_y);
    cudaFree(d_eps);
    cudaFree(d_eps_prime);
    cudaFree(d_eps2_x);
    cudaFree(d_eps2_y);
    cudaFree(d_state);

    free(T);
    free(p);
    free(theta);
    free(p_x);
    free(p_y);
    free(eps);
    free(eps_prime);
    free(eps2_x);
    free(eps2_y);

    std::cout << "t=" << t << std::endl;
    return 0;
}
