# CUDA Batch Image Processor

This project demonstrates GPU-accelerated batch image processing using CUDA by applying a Gaussian blur filter to multiple images concurrently.

---

## Compilation
```bash
make
Usage
./image_processor --input_dir <path_to_images> --output_dir <path_to_output>
Example
./image_processor --input_dir ./data/images --output_dir ./data/output
Project Description
The application processes a directory of images on the GPU using CUDA. A custom CUDA kernel applies a 3×3 Gaussian blur to each image. The program was validated using batches of more than ten high-resolution (4K) images. Key insights from this project include the impact of grid and block configuration on performance and the importance of careful boundary handling within GPU kernels.

Code Description
The codebase is designed to maximize data-parallel execution on the GPU:

Key Components
Main Function

Parses command-line arguments (--input_dir and --output_dir) to determine input and output directories.

Verifies directory validity before execution.

Initiates batch processing through the processImages function.

processImages Function

Traverses the input directory using std::filesystem to identify supported image formats (.jpg, .png).

For each image:

Loads image data into host memory using OpenCV (imread).

Allocates CUDA device memory for input and output buffers.

Transfers image data from host to device.

Launches the CUDA Gaussian blur kernel.

Copies processed data back to the host and saves the result using OpenCV (imwrite).

gaussianBlurKernel (CUDA Kernel)

Implements a two-dimensional CUDA kernel where each thread processes a single pixel.

Applies a fixed 3×3 Gaussian convolution mask with normalized weights.

Processes RGB channels independently.

Includes boundary checks using clamping (min/max) to prevent out-of-bounds memory access.

Executed using a 16×16 thread block configuration.

Implementation Details
Parallelism: Pixel-level parallelism is achieved by mapping one CUDA thread to one image pixel.

Memory Management: Device memory is explicitly managed using cudaMalloc and cudaMemcpy.

Error Handling: Handles invalid directories and image loading failures gracefully.

Code Style: Maintains consistent naming conventions, modular structure, and inline documentation in line with standard C++ style guidelines.

