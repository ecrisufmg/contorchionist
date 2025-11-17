#ifndef CORE_AP_DCT_H
#define CORE_AP_DCT_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>

#include "core_util_tensors.h"

namespace contorchionist {
namespace core {
namespace ap_dct {

template<typename T = float>
class DCT {
public:
    static torch::Tensor createDctMatrix(int64_t n_coeffs, int64_t N, int type, const std::string& norm, const torch::Device& device) {
        torch::Tensor matrix = torch::empty({n_coeffs, N}, torch::kFloat64); // Use double for precision
        auto matrix_a = matrix.accessor<double, 2>();
        double pi = M_PI;

        for (int64_t k = 0; k < n_coeffs; ++k) {
            for (int64_t n = 0; n < N; ++n) {
                double val = 0.0;
                if (type == 1) { // DCT-I
                    if (N <= 1) {
                         val = 1.0;
                    } else {
                        val = std::cos(pi * k * n / (N - 1));
                    }
                } else if (type == 2) { // DCT-II
                    val = std::cos(pi * k * (2.0 * n + 1.0) / (2.0 * N));
                } else if (type == 3) { // DCT-III
                     val = std::cos(pi * (2.0 * k + 1.0) * n / (2.0 * N));
                } else if (type == 4) { // DCT-IV
                    val = std::cos(pi * (2.0 * k + 1.0) * (2.0 * n + 1.0) / (4.0 * N));
                }
                matrix_a[k][n] = val;
            }
        }

        if (norm == "ortho") {
            if (type == 1) { // DCT-I Ortho Normalization
                if (N > 1) {
                    // Access rows as tensors to use mul_
                    // k=0 row
                    if (n_coeffs > 0) { // Check if the 0-th coefficient is actually being computed
                        matrix.select(0, 0).mul_(std::sqrt(1.0 / (N - 1)));
                        matrix.select(0, 0).mul_(1.0/std::sqrt(2.0)); // Additional scaling for k=0
                    }

                    for (int64_t k_row = 1; k_row < n_coeffs; ++k_row) { // k=1 to n_coeffs-1 rows
                        if (k_row == N-1 && (N-1) >= 0) { // Special scaling for k=N-1 row, ensure N-1 is valid index
                            matrix.select(0, k_row).mul_(std::sqrt(1.0 / (N - 1)));
                            matrix.select(0, k_row).mul_(1.0/std::sqrt(2.0));
                        } else {
                            matrix.select(0, k_row).mul_(std::sqrt(2.0 / (N - 1)));
                        }
                    }
                } else { // N=1 case for DCT-I ortho
                     matrix_a[0][0] = 1.0;
                }
            } else if (type == 2) { // DCT-II Ortho Normalization
                matrix.slice(0, 0, 1).mul_(std::sqrt(1.0 / N)); // k = 0
                if (n_coeffs > 1) {
                    matrix.slice(0, 1, n_coeffs).mul_(std::sqrt(2.0 / N)); // k > 0
                }
            } else if (type == 3) { // DCT-III Ortho Normalization
                for (int64_t k = 0; k < n_coeffs; ++k) {
                    for (int64_t n = 0; n < N; ++n) {
                        double scale = (n == 0) ? std::sqrt(1.0 / N) : std::sqrt(2.0 / N);
                        matrix_a[k][n] *= scale;
                    }
                }
            } else if (type == 4) { // DCT-IV Ortho Normalization
                matrix.mul_(std::sqrt(2.0 / N));
            }
        }
        return matrix.to(device).to(torch::kFloat32); // Cast to float32 for output
    }

    static torch::Tensor dctType1(const torch::Tensor& x, int n_coeffs, const std::string& norm = "none", const torch::Device& device = torch::kCPU) {
        if (x.dim() == 0 || x.size(-1) == 0) {
            return torch::empty({0}, x.options());
        }
        if (x.size(-1) <= 1 && n_coeffs > 0) {
            if (n_coeffs == 0) return torch::empty(contorchionist::core::util_tensors::prepend_leading_dims(x, 0).sizes(), x.options());
            auto result_shape_tensor = contorchionist::core::util_tensors::prepend_leading_dims(x, n_coeffs);
            torch::Tensor result = torch::zeros(result_shape_tensor.sizes(), x.options());
            if (n_coeffs > 0 && x.size(-1) == 1) { // Ensure x is not empty and has 1 element
                 result.narrow(-1, 0, 1) = x.narrow(-1,0,1);
            }
            return result;
        }

        int64_t N = x.size(-1);
        if (n_coeffs < 0) n_coeffs = N;
        if (n_coeffs == 0) return torch::empty(contorchionist::core::util_tensors::prepend_leading_dims(x, 0).sizes(), x.options());
        // actual_n_coeffs_for_matrix determines the number of rows in the DCT matrix.
        // For DCT-I, the number of basis functions is N (k=0,...,N-1).
        // So, we can't request more than N unique DCT-I coefficients.
        // If n_coeffs > N, we compute N coefficients and then pad.
        int64_t num_coeffs_to_compute = std::min(static_cast<int64_t>(n_coeffs), N);


        // Create the DCT-I matrix directly
        torch::Tensor dct_matrix = torch::empty({num_coeffs_to_compute, N},
                                              device.has_index() ?
                                              torch::TensorOptions().device(device).dtype(torch::kFloat64) :
                                              torch::TensorOptions().dtype(torch::kFloat64));
        auto dct_matrix_a = dct_matrix.accessor<double, 2>();
        double pi = M_PI;

        // This part is reached only if N > 1 due to earlier checks.
        for (int64_t k_idx = 0; k_idx < num_coeffs_to_compute; ++k_idx) {
            for (int64_t n_idx = 0; n_idx < N; ++n_idx) {
                dct_matrix_a[k_idx][n_idx] = std::cos(pi * k_idx * n_idx / (N - 1));
            }
        }

        if (norm == "ortho") { // N > 1 is implicit here
            dct_matrix.mul_(std::sqrt(2.0 / (N - 1)));
            if (num_coeffs_to_compute > 0) { // k=0 row
                dct_matrix.select(0, 0).mul_(1.0 / std::sqrt(2.0));
            }
            // k=N-1 row (if N-1 is a valid row index in our matrix)
            if (N - 1 < num_coeffs_to_compute) {
                dct_matrix.select(0, N - 1).mul_(1.0 / std::sqrt(2.0));
            }
        }
        dct_matrix = dct_matrix.to(x.scalar_type());

        torch::Tensor y = torch::matmul(x.unsqueeze(-2), dct_matrix.transpose(0, 1)).squeeze(-2);

        if (n_coeffs > N) {
            auto result_shape_tensor = contorchionist::core::util_tensors::prepend_leading_dims(x, n_coeffs);
            torch::Tensor padded_y = torch::zeros(result_shape_tensor.sizes(), y.options());
            padded_y.narrow(-1, 0, N) = y; // y contains N coefficients
            return padded_y;
        }
        // If n_coeffs <= N, y already has num_coeffs_to_compute (which is n_coeffs) elements.
        return y;
    }

    static torch::Tensor dctType2(const torch::Tensor& x, int n_coeffs, const std::string& norm = "none", const torch::Device& device = torch::kCPU) {
        // Check tensor validity first
        if (!x.defined()) {
            throw std::runtime_error("Input tensor is not defined");
        }

        if (x.dim() == 0 || x.size(-1) == 0) {
            return torch::empty({0}, x.options());
        }

        int64_t N = x.size(-1);
        if (n_coeffs < 0) n_coeffs = N;
        if (n_coeffs == 0) return torch::empty(contorchionist::core::util_tensors::prepend_leading_dims(x, 0).sizes(), x.options());

        try {
            torch::Tensor dct_matrix = createDctMatrix(n_coeffs, N, 2, norm, device);
            torch::Tensor y = torch::matmul(x.unsqueeze(-2), dct_matrix.transpose(0, 1)).squeeze(-2);

            if (norm != "ortho") {
                y.mul_(2.0);
            }
            return y;
        } catch (const std::exception& e) {
            throw std::runtime_error("Error in DCT Type 2 computation: " + std::string(e.what()));
        }
    }

    static torch::Tensor dctType3(const torch::Tensor& x, int n_coeffs, const std::string& norm = "none", const torch::Device& device = torch::kCPU) {
        if (x.dim() == 0 || x.size(-1) == 0) {
            return torch::empty({0}, x.options());
        }
        int64_t K_coeffs_in = x.size(-1);
        int64_t N_samples_out = n_coeffs;

        if (N_samples_out < 0) N_samples_out = K_coeffs_in;
        if (N_samples_out == 0) return torch::empty(contorchionist::core::util_tensors::prepend_leading_dims(x, 0).sizes(), x.options());

        torch::Tensor dct_matrix = createDctMatrix(N_samples_out, K_coeffs_in, 3, norm, device);

        torch::Tensor x_modified = x.clone();
        if (norm == "ortho" && K_coeffs_in > 0) {
            x_modified.narrow(-1, 0, 1).mul_(1.0 / std::sqrt(2.0));
        }

        torch::Tensor y = torch::matmul(x_modified.unsqueeze(-2), dct_matrix.transpose(0, 1)).squeeze(-2);

        if (norm != "ortho") {
            y.mul_(2.0);
        }
        return y;
    }

    static torch::Tensor dctType4(const torch::Tensor& x, int n_coeffs, const std::string& norm = "none", const torch::Device& device = torch::kCPU) {
        if (x.dim() == 0 || x.size(-1) == 0) {
            return torch::empty({0}, x.options());
        }
        int64_t N = x.size(-1);
        if (n_coeffs < 0) n_coeffs = N;
        if (n_coeffs == 0) return torch::empty(contorchionist::core::util_tensors::prepend_leading_dims(x, 0).sizes(), x.options());

        torch::Tensor dct_matrix = createDctMatrix(n_coeffs, N, 4, norm, device);
        torch::Tensor y = torch::matmul(x.unsqueeze(-2), dct_matrix.transpose(0, 1)).squeeze(-2);

        return y;
    }
};

} // namespace ap_dct
} // namespace core
} // namespace contorchionist

#endif // CORE_AP_DCT_H
