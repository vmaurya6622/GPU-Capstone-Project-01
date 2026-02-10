# CUDA Batch Image Processor
This project applies a Gaussian blur to multiple images in parallel using CUDA.


## Compilation
make

## Usage
./image_processor --input_dir <path_to_images> --output_dir <path_to_output>

## Example
./image_processor --input_dir ./data/images --output_dir ./data/output

## Project Description
This program processes a batch of images (tested with 10+ large 4K images) using a CUDA kernel to apply a 3x3 Gaussian blur. The CLI takes input/output directory paths as arguments. Lessons learned: Optimizing grid/block sizes improves performance; handling edge cases in the kernel is crucial.

## Code Description
The codebase is structured to leverage CUDA for parallel image processing:

### Key Components
1. **Main Function**:
   - Parses command-line arguments (`--input_dir` and `--output_dir`) to specify input/output directories.
   - Validates directory existence before processing.
   - Calls `processImages` to handle the batch.

2. **processImages Function**:
   - Iterates over all `.jpg` and `.png` files in the input directory using `std::filesystem`.
   - For each image:
     - Loads it using OpenCV (`imread`).
     - Allocates CUDA device memory for input/output data.
     - Copies image data to the GPU.
     - Launches the Gaussian blur kernel.
     - Copies the result back to the host and saves it with OpenCV (`imwrite`).

3. **gaussianBlurKernel (CUDA Kernel)**:
   - Implements a two-dimensional CUDA kernel where each thread processes a single pixel.
   - Applies a fixed 3×3 Gaussian convolution mask with normalized weights.
   - Processes RGB channels independently.
   - Includes boundary checks using clamping (min/max) to prevent out-of-bounds memory access.
   - Executed using a 16×16 thread block configuration.

### Implementation Details
- **Parallelism**: Each thread processes one pixel, with the grid sized dynamically based on image dimensions.
- **Memory Management**: Explicit CUDA memory allocation (`cudaMalloc`) and transfer (`cudaMemcpy`) for efficiency.
- **Error Handling**: Checks for file loading failures and directory existence.
- **Style**: Follows Google C++ Style Guide (consistent naming, comments, modularity).
