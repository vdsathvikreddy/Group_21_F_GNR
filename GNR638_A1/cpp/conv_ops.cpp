// conv_ops.cpp - Optimized C++ implementation of convolution operations
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <thread>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

// Helper function to convert nested Python list to flat vector
std::vector<float> flatten_4d(const std::vector<std::vector<std::vector<std::vector<float>>>>& data) {
    std::vector<float> result;
    for (const auto& b : data) {
        for (const auto& c : b) {
            for (const auto& h : c) {
                for (const auto& w : h) {
                    result.push_back(w);
                }
            }
        }
    }
    return result;
}

// Fast convolution forward pass with multi-threading
std::vector<std::vector<std::vector<std::vector<float>>>> conv2d_forward(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& weight,
    const std::vector<float>& bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w) {
    
    int batch_size = input.size();
    int in_channels = input[0].size();
    int input_h = input[0][0].size();
    int input_w = input[0][0][0].size();
    
    int out_channels = weight.size();
    int kernel_h = weight[0][0].size();
    int kernel_w = weight[0][0][0].size();
    
    // Calculate output dimensions
    int padded_h = input_h + 2 * padding_h;
    int padded_w = input_w + 2 * padding_w;
    int output_h = (padded_h - kernel_h) / stride_h + 1;
    int output_w = (padded_w - kernel_w) / stride_w + 1;
    
    // Initialize output
    std::vector<std::vector<std::vector<std::vector<float>>>> output(
        batch_size,
        std::vector<std::vector<std::vector<float>>>(
            out_channels,
            std::vector<std::vector<float>>(
                output_h,
                std::vector<float>(output_w, 0.0f)
            )
        )
    );
    
    // Add padding to input
    std::vector<std::vector<std::vector<std::vector<float>>>> padded_input(
        batch_size,
        std::vector<std::vector<std::vector<float>>>(
            in_channels,
            std::vector<std::vector<float>>(
                padded_h,
                std::vector<float>(padded_w, 0.0f)
            )
        )
    );
    
    // Copy input to padded array
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < input_h; ++h) {
                for (int w = 0; w < input_w; ++w) {
                    padded_input[b][c][h + padding_h][w + padding_w] = input[b][c][h][w];
                }
            }
        }
    }
    
    // Parallel convolution using threads
    auto conv_batch = [&](int batch_start, int batch_end) {
        for (int b = batch_start; b < batch_end; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow < output_w; ++ow) {
                        float sum = 0.0f;
                        int h_start = oh * stride_h;
                        int w_start = ow * stride_w;
                        
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    sum += padded_input[b][ic][h_start + kh][w_start + kw] 
                                         * weight[oc][ic][kh][kw];
                                }
                            }
                        }
                        
                        if (!bias.empty()) {
                            sum += bias[oc];
                        }
                        
                        output[b][oc][oh][ow] = sum;
                    }
                }
            }
        }
    };
    
    // Use multiple threads
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    int batch_per_thread = (batch_size + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
        int start = t * batch_per_thread;
        int end = std::min(start + batch_per_thread, batch_size);
        if (start < batch_size) {
            threads.emplace_back(conv_batch, start, end);
        }
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return output;
}

// Fast matrix multiplication for linear layers
std::vector<std::vector<float>> matmul(
    const std::vector<std::vector<float>>& a,
    const std::vector<std::vector<float>>& b) {
    
    int m = a.size();
    int k = a[0].size();
    int n = b[0].size();
    
    std::vector<std::vector<float>> result(m, std::vector<float>(n, 0.0f));
    
    // Parallel matrix multiplication
    auto matmul_rows = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < k; ++p) {
                    sum += a[i][p] * b[p][j];
                }
                result[i][j] = sum;
            }
        }
    };
    
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    int rows_per_thread = (m + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
        int start = t * rows_per_thread;
        int end = std::min(start + rows_per_thread, m);
        if (start < m) {
            threads.emplace_back(matmul_rows, start, end);
        }
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return result;
}

// Fast ReLU activation
std::vector<std::vector<std::vector<std::vector<float>>>> relu_forward(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input) {
    
    auto output = input;
    
    for (auto& b : output) {
        for (auto& c : b) {
            for (auto& h : c) {
                for (auto& val : h) {
                    val = std::max(0.0f, val);
                }
            }
        }
    }
    
    return output;
}

// Fast max pooling
std::vector<std::vector<std::vector<std::vector<float>>>> maxpool2d_forward(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w) {
    
    int batch_size = input.size();
    int channels = input[0].size();
    int input_h = input[0][0].size();
    int input_w = input[0][0][0].size();
    
    int output_h = (input_h - kernel_h) / stride_h + 1;
    int output_w = (input_w - kernel_w) / stride_w + 1;
    
    std::vector<std::vector<std::vector<std::vector<float>>>> output(
        batch_size,
        std::vector<std::vector<std::vector<float>>>(
            channels,
            std::vector<std::vector<float>>(
                output_h,
                std::vector<float>(output_w, 0.0f)
            )
        )
    );
    
    // Parallel max pooling
    auto pool_batch = [&](int batch_start, int batch_end) {
        for (int b = batch_start; b < batch_end; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow < output_w; ++ow) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        int h_start = oh * stride_h;
                        int w_start = ow * stride_w;
                        
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                max_val = std::max(max_val, 
                                    input[b][c][h_start + kh][w_start + kw]);
                            }
                        }
                        
                        output[b][c][oh][ow] = max_val;
                    }
                }
            }
        }
    };
    
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    int batch_per_thread = (batch_size + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
        int start = t * batch_per_thread;
        int end = std::min(start + batch_per_thread, batch_size);
        if (start < batch_size) {
            threads.emplace_back(pool_batch, start, end);
        }
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return output;
}

PYBIND11_MODULE(conv_ops, m) {
    m.doc() = "Fast C++ convolution and linear operations";
    
    m.def("conv2d_forward", &conv2d_forward,
          "Fast convolution forward pass",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride_h"),
          py::arg("stride_w"),
          py::arg("padding_h"),
          py::arg("padding_w"));
    
    m.def("matmul", &matmul,
          "Fast matrix multiplication",
          py::arg("a"),
          py::arg("b"));
    
    m.def("relu_forward", &relu_forward,
          "Fast ReLU activation",
          py::arg("input"));
    
    m.def("maxpool2d_forward", &maxpool2d_forward,
          "Fast max pooling",
          py::arg("input"),
          py::arg("kernel_h"),
          py::arg("kernel_w"),
          py::arg("stride_h"),
          py::arg("stride_w"));
}
