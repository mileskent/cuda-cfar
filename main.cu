#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <matplot/matplot.h>

const int ARRAY_SIZE = 500;
const float SAMPLE_RATE = 10.0f;
const int FFT_PADDING_FACTOR = 4;
const int FFT_LEN = ARRAY_SIZE * FFT_PADDING_FACTOR;

// Signal Parameters
const float F1 = 1.5f; // Hz
const float F2 = 2.7f; // Hz
const float F3 = 3.7f; // Hz
const float POWER_NORM_1_DB = -6.0f; // dB
const float POWER_NORM_2_DB = -9.0f; // dB
const float POWER_NORM_3_DB = 0.0f; // dB

// Noise Parameters (from previous step)
const float NOISE_MU = 0.0f;
const float NOISE_SIGMA_DB = -10.0f; // dB

// Helper Constant for 2*pi
const float TWO_PI = 6.28318530718f;

/**
 * @brief CUDA kernel to compute the normalized magnitude spectrum in dB.
 * * This kernel performs: |X_k| = sqrt(real^2 + imag^2)
 * Normalized |X_k| /= N / 2
 * Log scaled: 10 * log10(|X_k|)
 * * @param d_input_complex Input: Pointer to the complex FFT output array on device.
 * @param d_output_real Output: Pointer to the final real-valued magnitude (dB) array on device.
 * @param N_orig The original length of the signal (for normalization).
 * @param fft_len The length of the FFT array.
 */
__global__ void postProcessSpectrumKernel(
    const cuFloatComplex* d_input_complex, 
    float* d_output_real, 
    int N_orig, 
    int fft_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < fft_len) {
        // 1. Get the complex component
        float real = d_input_complex[idx].x;
        float imag = d_input_complex[idx].y;

        // 2. Magnitude Calculation: |X_k| = sqrt(real^2 + imag^2)
        float magnitude = sqrtf(real * real + imag * imag);

        // 3. Normalization: |X_k| /= N / 2
        float normalized_magnitude = magnitude / (N_orig / 2.0f);
        
        // Add a small epsilon to prevent log(0)
        const float EPSILON = 1e-6f;
        float safe_magnitude = max(normalized_magnitude, EPSILON);

        // 4. Log Scaling (dB): 10 * log10(|X_k|)
        // Note: logf() uses natural log (ln). We need log10(). 
        // log10(x) = log(x) / log(10)
        float value_db = 10.0f * (logf(safe_magnitude) / logf(10.0f));

        d_output_real[idx] = value_db;
    }
}

/**
 * @brief Simple CUDA kernel to process an array on the GPU.
 * * For this example, it simply reads the data and writes it back,
 * effectively doing nothing to the values.
 * * @param d_array Pointer to the array data on the device.
 * @param array_size The number of elements in the array.
 */
__global__ void processArrayKernel(float* d_array, int array_size) {
    // Calculate the global index for the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds to ensure the thread is operating within the array size
    if (idx < array_size) {
        // Read the value, and write it back. No actual computation needed for this request.
        d_array[idx] = d_array[idx]; 
    }
}

// Calculates the linear standard deviation (sigma)
float get_linear_sigma(float db) {
    return std::pow(10.0f, db / 10.0f);
}

/**
 * @brief Performs the FFT analysis pipeline on the signal array using cuFFT and a custom CUDA kernel.
 * * Steps performed:
 * 1. Host: Apply Hanning Window and Zero-pad the signal.
 * 2. Device: cuFFT (C2C) transform.
 * 3. Device: Post-Processing Kernel (Magnitude, Normalization, Log-Scaling to dB).
 * 4. Host: Circular shift (fftshift) for plotting.
 * * @param h_signal_in The host array containing the time-domain signal (x_n).
 * @param sample_rate The sampling rate (fs). (Not directly used in this function, but needed for plotting context)
 * @return std::vector<float> A vector containing the final fftshifted 10*log10(|X_k|) values.
 */
std::vector<float> performFFT(const std::vector<float>& h_signal_in, float sample_rate) {
    const int N = h_signal_in.size();
    const int FFT_LEN = N * FFT_PADDING_FACTOR; // Assumes FFT_PADDING_FACTOR is defined (e.g., 4)
    const int OUTPUT_MEM_SIZE = FFT_LEN * sizeof(float);

    // --- 1. Host: Prepare Window and Output Vectors ---
    std::vector<float> h_windowed_signal(FFT_LEN, 0.0f); 

    // Apply Hanning Window and Zero-Padding
    for (int i = 0; i < N; ++i) {
        // Hanning Window: 0.5 * (1 - cos(2*pi*i / (N-1)))
        float window_val = 0.5f * (1.0f - std::cos(TWO_PI * (float)i / (N - 1)));
        h_windowed_signal[i] = h_signal_in[i] * window_val;
    }
    
    // --- 2. Device: Allocate Memory and Copy Input ---
    
    // d_fft_inout is the buffer for both the complex input (real signal + zero imag) and the complex output.
    cuFloatComplex* d_fft_inout = nullptr;
    // d_fft_log_mag will hold the final real-valued magnitude spectrum in dB.
    float* d_fft_log_mag = nullptr;

    cudaMalloc((void**)&d_fft_inout, FFT_LEN * sizeof(cuFloatComplex));
    cudaMalloc((void**)&d_fft_log_mag, OUTPUT_MEM_SIZE);
    
    // Copy the real signal data to the complex device input buffer
    std::vector<cuFloatComplex> h_initial_complex(FFT_LEN);
    for (int i = 0; i < FFT_LEN; ++i) {
        h_initial_complex[i].x = h_windowed_signal[i]; // Real part (signal or padding zero)
        h_initial_complex[i].y = 0.0f;                 // Imaginary part (zero)
    }
    cudaMemcpy(d_fft_inout, h_initial_complex.data(), FFT_LEN * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    // --- 3. cuFFT Plan and Execution ---
    cufftHandle plan;
    // CUFFT_C2C: Complex-to-Complex transform (since we treated the real input as complex with zero imaginary part)
    cufftPlan1d(&plan, FFT_LEN, CUFFT_C2C, 1); 

    cufftExecC2C(plan, d_fft_inout, d_fft_inout, CUFFT_FORWARD);
    cudaDeviceSynchronize(); 

    // Check cuFFT errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuFFT execution failed: " << cudaGetErrorString(err) << std::endl;
        // Need to clean up before exiting
        cufftDestroy(plan);
        cudaFree(d_fft_inout);
        cudaFree(d_fft_log_mag);
        return {}; 
    }

    // --- 4. Launch Post-Processing Kernel (on GPU) ---
    // This kernel computes |X_k|, normalizes, and calculates 10*log10(|X_k|)
    const int THREADS_PER_BLOCK = 256;
    int num_blocks = (FFT_LEN + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    postProcessSpectrumKernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_fft_inout,      // Input: Complex FFT result
        d_fft_log_mag,    // Output: Real-valued dB magnitude
        N,                // Original array size for normalization
        FFT_LEN           // Length of the FFT array
    );
    cudaDeviceSynchronize(); 

    // Check for errors after the kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Post-processing kernel failed: " << cudaGetErrorString(err) << std::endl;
    }

    // --- 5. Copy Final Result Back to Host (DtoH) ---
    std::vector<float> h_Xk_log(FFT_LEN);
    // Copy the real-valued dB spectrum from the device
    cudaMemcpy(h_Xk_log.data(), d_fft_log_mag, OUTPUT_MEM_SIZE, cudaMemcpyDeviceToHost);

    // --- 6. Host: Perform fftshift ---
    // Shift the DC component (0 Hz) to the center for plotting
    int shift = FFT_LEN / 2;
    std::vector<float> h_shifted_output(FFT_LEN);
    for (int i = 0; i < FFT_LEN; ++i) {
        int shifted_idx = (i + shift) % FFT_LEN;
        h_shifted_output[i] = h_Xk_log[shifted_idx];
    }
    
    // --- 7. Cleanup ---
    cufftDestroy(plan);
    cudaFree(d_fft_inout);
    cudaFree(d_fft_log_mag);

    return h_shifted_output;
}



/**
 * @brief Generates an array of random float noise following a Normal (Gaussian) distribution.
 * * @param size The desired length of the array.
 * @param sigma_linear The linear standard deviation.
 * @return std::vector<float> A vector containing random noise values.
 */
std::vector<float> generateGaussianNoise(int size, float sigma_linear) {
    std::vector<float> noise(size);
    // Using a fixed seed (0) for reproducibility
    std::default_random_engine generator(0); 
    // Normal Distribution: mean=0, std_dev=sigma_linear
    std::normal_distribution<float> distribution(NOISE_MU, sigma_linear);

    for (int i = 0; i < size; ++i) {
        noise[i] = distribution(generator);
    }
    return noise;
}


/**
 * @brief Generates the composite signal x_n using three sine waves and Gaussian noise.
 * * @param size The size of the signal array.
 * @param sample_rate The sampling rate for the time vector.
 * @return std::vector<float> The final composite signal array.
 */
 std::vector<float> generateCompositeSignal(int size, float sample_rate) {
    // 1. Calculate Linear Amplitudes and Noise Sigma
    const float A1 = get_linear_sigma(POWER_NORM_1_DB);
    const float A2 = get_linear_sigma(POWER_NORM_2_DB);
    const float A3 = get_linear_sigma(POWER_NORM_3_DB);
    const float noise_sigma_linear = get_linear_sigma(NOISE_SIGMA_DB);
    
    // Denominator for normalization
    const float denominator = A1 + A2 + A3 + noise_sigma_linear;

    // 2. Generate Time Vector (t)
    std::vector<float> t(size);
    for (int i = 0; i < size; ++i) {
        // t[i] = i * (1 / sample_rate)
        t[i] = (float)i / sample_rate;
    }

    // 3. Generate Noise Vector
    std::vector<float> noise = generateGaussianNoise(size, noise_sigma_linear);

    // 4. Calculate Composite Signal (x_n)
    std::vector<float> x_n(size);
    for (int i = 0; i < size; ++i) {
        // Sum of sine waves
        float signal_sum = 
            A1 * std::sin(TWO_PI * F1 * t[i]) +
            A2 * std::sin(TWO_PI * F2 * t[i]) +
            A3 * std::sin(TWO_PI * F3 * t[i]);
        
        // Final signal calculation
        x_n[i] = (signal_sum + noise[i]) / denominator;
    }
    
    std::cout << "Signal generated. Denominator for normalization: " << denominator << std::endl;
    return x_n;
}

int main() {
    std::vector<float> h_signal_in = generateCompositeSignal(ARRAY_SIZE, SAMPLE_RATE);
    
    std::cout << "Starting FFT Analysis Pipeline..." << std::endl;
    std::cout << "---" << std::endl;

    std::vector<float> h_fft_spectrum = performFFT(h_signal_in, SAMPLE_RATE);
    
    std::cout << "FFT computation complete and data transferred back." << std::endl;
    std::cout << "---" << std::endl;

    // Generate Frequency Vector (X-axis for the spectrum plot)
    const int POS_FFT_LEN = FFT_LEN / 2;
    std::vector<double> freq_data(POS_FFT_LEN);
    float freq_step = SAMPLE_RATE / FFT_LEN;
    float start_freq = 0;
    
    for (int i = 0; i < POS_FFT_LEN; ++i) {
        freq_data[i] = start_freq + i * freq_step;
    }

    // Convert the FFT spectrum to double for Matplot++ plotting
    std::vector<double> spectrum_y_data(h_fft_spectrum.begin() + FFT_LEN / 2, h_fft_spectrum.end());
    
    matplot::figure();
    matplot::plot(freq_data, spectrum_y_data, "r-");
    
    matplot::title("Frequency Spectrum (FFT)");
    matplot::xlabel("Frequency (Hz)");
    matplot::ylabel("Magnitude (dB)");
    matplot::grid(matplot::on);
    matplot::show(); 
    
    std::cout << "Program terminated." << std::endl;

    return 0;
}