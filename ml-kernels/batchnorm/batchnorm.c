/*
 * Batch Normalization Kernel for CARTS
 *
 * Extracted and adapted from Darknet framework
 * Original: https://github.com/pjreddie/darknet
 *
 * Description:
 *   Batch normalization normalizes activations across the batch dimension.
 *   For each channel, computes mean and variance across batch and spatial dimensions,
 *   then normalizes, scales, and shifts the activations.
 *
 * Algorithm:
 *   For each channel c:
 *     1. Compute mean: μ_c = mean(x[b,c,h,w] for all b,h,w)
 *     2. Compute variance: σ²_c = var(x[b,c,h,w] for all b,h,w)
 *     3. Normalize: x̂[b,c,h,w] = (x[b,c,h,w] - μ_c) / sqrt(σ²_c + ε)
 *     4. Scale and shift: y[b,c,h,w] = γ_c * x̂[b,c,h,w] + β_c
 *
 * Memory Layout: NCHW (batch, channels, height, width)
 * Index calculation: index = b*filters*spatial + f*spatial + i
 * where spatial = height * width
 *
 * CARTS Compatibility:
 * - No global variables
 * - Clean parameter passing
 * - OpenMP parallelization
 * - Self-contained functions
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Problem size configuration
#ifndef BATCH_SIZE
#define BATCH_SIZE 4
#endif

#ifndef CHANNELS
#define CHANNELS 64
#endif

#ifndef HEIGHT
#define HEIGHT 32
#endif

#ifndef WIDTH
#define WIDTH 32
#endif

#define EPSILON 0.00001f

/*
 * Compute mean across batch and spatial dimensions for each channel
 *
 * Parameters:
 *   x: Input tensor [batch, filters, spatial]
 *   batch: Batch size
 *   filters: Number of channels
 *   spatial: Height * Width
 *   mean: Output mean for each channel [filters]
 */
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1.0f / (batch * spatial);
    int i, j, k;

    #ifdef _OPENMP
    #pragma omp parallel for private(j, k)
    #endif
    for (i = 0; i < filters; ++i) {
        mean[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j * filters * spatial + i * spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

/*
 * Compute variance across batch and spatial dimensions for each channel
 *
 * Parameters:
 *   x: Input tensor [batch, filters, spatial]
 *   mean: Mean for each channel [filters]
 *   batch: Batch size
 *   filters: Number of channels
 *   spatial: Height * Width
 *   variance: Output variance for each channel [filters]
 */
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1.0f / (batch * spatial - 1);
    int i, j, k;

    #ifdef _OPENMP
    #pragma omp parallel for private(j, k)
    #endif
    for (i = 0; i < filters; ++i) {
        variance[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j * filters * spatial + i * spatial + k;
                float diff = x[index] - mean[i];
                variance[i] += diff * diff;
            }
        }
        variance[i] *= scale;
    }
}

/*
 * Normalize activations using mean and variance
 *
 * Parameters:
 *   x: Input/output tensor [batch, filters, spatial] (modified in-place)
 *   mean: Mean for each channel [filters]
 *   variance: Variance for each channel [filters]
 *   batch: Batch size
 *   filters: Number of channels
 *   spatial: Height * Width
 */
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;

    #ifdef _OPENMP
    #pragma omp parallel for private(f, i) collapse(2)
    #endif
    for (b = 0; b < batch; ++b) {
        for (f = 0; f < filters; ++f) {
            for (i = 0; i < spatial; ++i) {
                int index = b * filters * spatial + f * spatial + i;
                x[index] = (x[index] - mean[f]) / (sqrtf(variance[f] + EPSILON));
            }
        }
    }
}

/*
 * Scale normalized activations by learned scale parameters
 *
 * Parameters:
 *   output: Output tensor [batch, filters, spatial] (modified in-place)
 *   scales: Scale parameter for each channel [filters]
 *   batch: Batch size
 *   n: Number of channels
 *   size: Spatial size (height * width)
 */
void scale_bias_cpu(float *output, float *scales, int batch, int n, int size)
{
    int i, j, b;

    #ifdef _OPENMP
    #pragma omp parallel for private(i, j) collapse(2)
    #endif
    for (b = 0; b < batch; ++b) {
        for (i = 0; i < n; ++i) {
            for (j = 0; j < size; ++j) {
                int index = b * n * size + i * size + j;
                output[index] *= scales[i];
            }
        }
    }
}

/*
 * Add learned bias to scaled activations
 *
 * Parameters:
 *   output: Output tensor [batch, filters, spatial] (modified in-place)
 *   biases: Bias parameter for each channel [filters]
 *   batch: Batch size
 *   n: Number of channels
 *   size: Spatial size (height * width)
 */
void add_bias_cpu(float *output, float *biases, int batch, int n, int size)
{
    int i, j, b;

    #ifdef _OPENMP
    #pragma omp parallel for private(i, j) collapse(2)
    #endif
    for (b = 0; b < batch; ++b) {
        for (i = 0; i < n; ++i) {
            for (j = 0; j < size; ++j) {
                int index = b * n * size + i * size + j;
                output[index] += biases[i];
            }
        }
    }
}

/*
 * Complete batch normalization forward pass
 *
 * Parameters:
 *   x: Input tensor [batch, channels, height, width]
 *   output: Output tensor [batch, channels, height, width]
 *   scales: Scale parameters γ [channels]
 *   biases: Bias parameters β [channels]
 *   batch: Batch size
 *   channels: Number of channels
 *   height: Spatial height
 *   width: Spatial width
 *   mean: Workspace for mean [channels]
 *   variance: Workspace for variance [channels]
 */
void batchnorm_forward(
    float *x,
    float *output,
    float *scales,
    float *biases,
    int batch,
    int channels,
    int height,
    int width,
    float *mean,
    float *variance)
{
    int spatial = height * width;
    int i;

    // Copy input to output (will be modified in-place)
    int total_size = batch * channels * spatial;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (i = 0; i < total_size; ++i) {
        output[i] = x[i];
    }

    // Compute statistics
    mean_cpu(output, batch, channels, spatial, mean);
    variance_cpu(output, mean, batch, channels, spatial, variance);

    // Normalize
    normalize_cpu(output, mean, variance, batch, channels, spatial);

    // Scale and shift
    scale_bias_cpu(output, scales, batch, channels, spatial);
    add_bias_cpu(output, biases, batch, channels, spatial);
}

/*
 * Initialize test data
 */
void init_data(float *x, float *scales, float *biases, int batch, int channels, int spatial)
{
    int i;
    int total = batch * channels * spatial;

    // Initialize input with some pattern
    for (i = 0; i < total; ++i) {
        x[i] = ((float)i / total) * 2.0f - 1.0f;  // Range [-1, 1]
    }

    // Initialize scales to 1.0 and biases to 0.0
    for (i = 0; i < channels; ++i) {
        scales[i] = 1.0f;
        biases[i] = 0.0f;
    }
}

/*
 * Compute reference statistics for validation
 */
void compute_reference_stats(float *output, int batch, int channels, int spatial,
                             float *ref_mean, float *ref_variance)
{
    // Compute mean and variance of normalized output
    mean_cpu(output, batch, channels, spatial, ref_mean);
    variance_cpu(output, ref_mean, batch, channels, spatial, ref_variance);
}

int main(int argc, char **argv)
{
    int batch = BATCH_SIZE;
    int channels = CHANNELS;
    int height = HEIGHT;
    int width = WIDTH;
    int spatial = height * width;
    int total_size = batch * channels * spatial;

    printf("Batch Normalization Kernel\n");
    printf("===========================\n");
    printf("Configuration:\n");
    printf("  Batch size: %d\n", batch);
    printf("  Channels: %d\n", channels);
    printf("  Spatial: %d x %d = %d\n", height, width, spatial);
    printf("  Total elements: %d\n", total_size);
    printf("\n");

    // Allocate memory
    float *x = (float *)malloc(total_size * sizeof(float));
    float *output = (float *)malloc(total_size * sizeof(float));
    float *scales = (float *)malloc(channels * sizeof(float));
    float *biases = (float *)malloc(channels * sizeof(float));
    float *mean = (float *)malloc(channels * sizeof(float));
    float *variance = (float *)malloc(channels * sizeof(float));

    if (!x || !output || !scales || !biases || !mean || !variance) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize data
    init_data(x, scales, biases, batch, channels, spatial);

    // Run batch normalization
    printf("Running batch normalization...\n");
    batchnorm_forward(x, output, scales, biases, batch, channels, height, width, mean, variance);

    // Validation: normalized output should have mean ≈ 0 and variance ≈ 1 (before scale/bias)
    float *ref_mean = (float *)malloc(channels * sizeof(float));
    float *ref_variance = (float *)malloc(channels * sizeof(float));

    // Check a few statistics
    printf("\nValidation:\n");
    printf("  First 5 channel means: ");
    for (int i = 0; i < 5 && i < channels; ++i) {
        printf("%.6f ", mean[i]);
    }
    printf("\n");

    printf("  First 5 channel variances: ");
    for (int i = 0; i < 5 && i < channels; ++i) {
        printf("%.6f ", variance[i]);
    }
    printf("\n");

    printf("  First 5 output values: ");
    for (int i = 0; i < 5; ++i) {
        printf("%.6f ", output[i]);
    }
    printf("\n");

    printf("\nBatch normalization completed successfully!\n");

    // Cleanup
    free(x);
    free(output);
    free(scales);
    free(biases);
    free(mean);
    free(variance);
    free(ref_mean);
    free(ref_variance);

    return 0;
}
