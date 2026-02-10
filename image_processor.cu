#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <ctime>

using namespace cv;
namespace fs = std::filesystem;

// CUDA kernel for 2D Gaussian blur (simplified 3x3 kernel)
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float kernel[3][3] = {
        {1.0/16, 2.0/16, 1.0/16},
        {2.0/16, 4.0/16, 2.0/16},
        {1.0/16, 2.0/16, 1.0/16}
    };

    for (int c = 0; c < channels; c++) {
        float sum = 0.0;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                sum += input[(py * width + px) * channels + c] * kernel[ky + 1][kx + 1];
            }
        }
        output[(y * width + x) * channels + c] = (unsigned char)sum;
    }
}

// Get current timestamp for logging
std::string getTimestamp() {
    time_t now = time(0);
    char* dt = ctime(&now);
    std::string timestamp(dt);
    return timestamp.substr(0, timestamp.length() - 1); // Remove newline
}

// Process a batch of images and log results
void processImages(const std::string& inputDir, const std::string& outputDir, std::ofstream& logFile) {
    std::vector<std::string> imageFiles;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            imageFiles.push_back(entry.path().string());
        }
    }

    if (imageFiles.empty()) {
        std::cerr << "No images found in " << inputDir << std::endl;
        logFile << "[" << getTimestamp() << "] ERROR: No images found in " << inputDir << std::endl;
        return;
    }

    logFile << "[" << getTimestamp() << "] INFO: Starting batch processing of " << imageFiles.size() << " images" << std::endl;

    for (const auto& file : imageFiles) {
        // Load image
        Mat img = imread(file, IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Failed to load " << file << std::endl;
            logFile << "[" << getTimestamp() << "] ERROR: Failed to load " << file << std::endl;
            continue;
        }

        int width = img.cols;
        int height = img.rows;
        int channels = img.channels();
        size_t size = width * height * channels;

        // Allocate device memory
        unsigned char *d_input, *d_output;
        cudaMalloc(&d_input, size);
        cudaMalloc(&d_output, size);

        // Copy image data to device
        cudaMemcpy(d_input, img.data, size, cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

        // Copy result back to host
        Mat outputImg(height, width, CV_8UC3);
        cudaMemcpy(outputImg.data, d_output, size, cudaMemcpyDeviceToHost);

        // Save output
        std::string outputFile = outputDir + "/" + fs::path(file).filename().string();
        imwrite(outputFile, outputImg);

        std::cout << "Processed: " << file << " -> " << outputFile << std::endl;
        logFile << "[" << getTimestamp() << "] INFO: Processed " << file << " -> " << outputFile 
                << " (Size: " << width << "x" << height << ")" << std::endl;

        // Clean up
        cudaFree(d_input);
        cudaFree(d_output);
    }

    logFile << "[" << getTimestamp() << "] INFO: Batch processing completed" << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 5 || std::string(argv[1]) != "--input_dir" || std::string(argv[3]) != "--output_dir") {
        std::cerr << "Usage: " << argv[0] << " --input_dir <path> --output_dir <path>" << std::endl;
        return -1;
    }

    std::string inputDir = argv[2];
    std::string outputDir = argv[4];

    if (!fs::exists(inputDir) || !fs::exists(outputDir)) {
        std::cerr << "Input or output directory does not exist!" << std::endl;
        return -1;
    }

    // Open log file
    std::ofstream logFile("processing_log.txt");
    if (!logFile.is_open()) {
        std::cerr << "Failed to open log file!" << std::endl;
        return -1;
    }

    processImages(inputDir, outputDir, logFile);
    logFile.close();
    return 0;
}
