/***************************************************************************************************
 * from Cute tutorials
 **************************************************************************************************/

#pragma once
#include <cmath>
#include <cute/tensor.hpp>
#include <iostream>
#include <limits>

// Matrix norm calculation result
template <class T> struct MatrixNorm {
  T inf_norm;
  bool found_nan;

  MatrixNorm(T norm = T(0), bool nan = false)
      : inf_norm(norm), found_nan(nan) {}
};

// Calculate infinity norm of a matrix (max absolute value)
template <class Tensor> auto matrix_inf_norm(Tensor const &tensor) {
  using ValueType = typename Tensor::element_type;
  using FloatType =
      std::conditional_t<std::is_same_v<ValueType, cutlass::half_t>, float,
                         ValueType>;

  FloatType max_val = 0.0;
  bool found_nan = false;

  for (int i = 0; i < cute::size(tensor); ++i) {
    FloatType val = static_cast<FloatType>(tensor(i));
    if (std::isnan(val)) {
      found_nan = true;
    } else {
      max_val = std::max(max_val, std::abs(val));
    }
  }

  return MatrixNorm<FloatType>{max_val, found_nan};
}

// Calculate infinity norm of difference between two matrices
template <class TensorA, class TensorB>
auto matrix_diff_inf_norm(TensorA const &tensor_A, TensorB const &tensor_B) {
  using ValueTypeA = typename TensorA::element_type;
  using ValueTypeB = typename TensorB::element_type;
  using FloatType =
      std::conditional_t<std::is_same_v<ValueTypeA, cutlass::half_t> ||
                             std::is_same_v<ValueTypeB, cutlass::half_t>,
                         float, std::common_type_t<ValueTypeA, ValueTypeB>>;

  FloatType max_diff = 0.0;
  bool found_nan = false;

  for (int i = 0; i < cute::size(tensor_A); ++i) {
    FloatType val_A = static_cast<FloatType>(tensor_A(i));
    FloatType val_B = static_cast<FloatType>(tensor_B(i));
    FloatType diff = std::abs(val_A - val_B);

    if (std::isnan(diff)) {
      found_nan = true;
    } else {
      max_diff = std::max(max_diff, diff);
    }
  }

  return MatrixNorm<FloatType>{max_diff, found_nan};
}

template <class AccType, class TensorA, class TensorB, class TensorC,
          class TensorD, class Alpha, class Beta>
void reference_gemm(TensorA const &tensor_A, TensorB const &tensor_B,
                    TensorC const &tensor_C, TensorD &tensor_D, Alpha alpha,
                    Beta beta) {
  using namespace cute;
  for (int m = 0; m < size<0>(tensor_D); ++m) {
    for (int n = 0; n < size<1>(tensor_D); ++n) {
      AccType c = AccType(0.f);
      for (int k = 0; k < size<1>(tensor_A); ++k) {
        c += static_cast<AccType>(tensor_A(m, k)) *
             static_cast<AccType>(tensor_B(n, k));
      }
      tensor_D(m, n) = static_cast<typename TensorD::element_type>(
          static_cast<AccType>(alpha) * c +
          static_cast<AccType>(beta) * static_cast<AccType>(tensor_C(m, n)));
    }
  }
}

template <class TensorA, class TensorB, class TensorC, class TensorD,
          class RefTensorD>
bool compare_results(TensorA const &tensor_A, TensorB const &tensor_B,
                     TensorC const &tensor_C, TensorD const &tensor_D,
                     RefTensorD const &ref_tensor_D, bool print_diff = false,
                     bool verbose = true) {
  using namespace cute;
  auto norm_A = matrix_inf_norm(tensor_A);
  auto norm_B = matrix_inf_norm(tensor_B);
  auto norm_C = matrix_inf_norm(tensor_C);
  auto norm_D = matrix_inf_norm(tensor_D);
  auto norm_ref_D = matrix_inf_norm(ref_tensor_D);
  auto norm_diff = matrix_diff_inf_norm(tensor_D, ref_tensor_D);

  if (print_diff) {
    std::cout << "Element-wise differences:" << std::endl;
    int print_count = 0;
    const int max_print = 20; // Limit output
    for (int m = 0; m < size<0>(tensor_D) && print_count < max_print; ++m) {
      for (int n = 0; n < size<1>(tensor_D) && print_count < max_print; ++n) {
        auto diff = std::abs(static_cast<float>(tensor_D(m, n)) -
                             static_cast<float>(ref_tensor_D(m, n)));
        if (diff > 1e-5) { // Only print significant differences
          std::cout << "  [" << m << "," << n
                    << "] : " << static_cast<float>(tensor_D(m, n)) << " vs. "
                    << static_cast<float>(ref_tensor_D(m, n))
                    << " (diff: " << diff << ")" << std::endl;
          print_count++;
        }
      }
    }
    if (print_count == max_print) {
      std::cout << "  ... (showing first " << max_print
                << " significant differences)" << std::endl;
    }
  }

  if (verbose) {
    std::cout << "Matrix norms:" << std::endl;
    std::cout << "  ||A||_inf       : " << norm_A.inf_norm << std::endl;
    std::cout << "  ||B||_inf       : " << norm_B.inf_norm << std::endl;
    std::cout << "  ||C||_inf       : " << norm_C.inf_norm << std::endl;
    std::cout << "  ||D||_inf       : " << norm_D.inf_norm << std::endl;
    std::cout << "  ||ref_D||_inf   : " << norm_ref_D.inf_norm << std::endl;
    std::cout << "  ||D-ref_D||_inf : " << norm_diff.inf_norm << std::endl;
  }

  // Check for NaNs
  bool has_nans = norm_A.found_nan || norm_B.found_nan || norm_C.found_nan ||
                  norm_D.found_nan || norm_ref_D.found_nan;

  // Check for non-zero norms (valid data)
  bool has_valid_data = (norm_A.inf_norm > 0.0) && (norm_B.inf_norm > 0.0) &&
                        (norm_C.inf_norm > 0.0) && (norm_D.inf_norm > 0.0) &&
                        (norm_ref_D.inf_norm > 0.0);

  // Use relative tolerance for better floating point comparison
  auto relative_error =
      norm_diff.inf_norm / std::max(norm_D.inf_norm, norm_ref_D.inf_norm);
  const auto tolerance = 1e-4; // Tolerance for FP16 -> FP32 mixed precision

  bool results_match =
      norm_diff.inf_norm <= tolerance || relative_error <= tolerance;

  if (verbose && !results_match) {
    std::cout << "  Relative error  : " << relative_error << std::endl;
    std::cout << "  Tolerance       : " << tolerance << std::endl;
  }

  return !has_nans && has_valid_data && results_match;
}

template <class Tensor>
void initialize_tensor(Tensor &tensor,
                       cute::tuple<int, int> value_range = {-2, 2}) {
  using DataType = typename Tensor::element_type;
  auto [min, max] = value_range;
  for (int i = 0; i < cute::size(tensor); i++) {
    tensor(i) = static_cast<DataType>(
        int((max - min) * (rand() / double(RAND_MAX)) + min));
  }
}
