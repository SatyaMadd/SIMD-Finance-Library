#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <memory>
#include <random>
#include <cassert>
#include <thread>
#include <cstring>

#ifdef __aarch64__
    #include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#endif


#ifdef __aarch64__
    #define SIMD_ALIGN 16
#else
    #define SIMD_ALIGN 32
#endif


template<typename T>
class aligned_allocator {
public:
    using value_type = T;
    
    aligned_allocator() = default;
    template<typename U>
    aligned_allocator(const aligned_allocator<U>&) {}
    
    T* allocate(std::size_t n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, SIMD_ALIGN, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* ptr, std::size_t) {
        free(ptr);
    }
};

template<typename T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;


inline void prefetch_read(const void* ptr) {
#ifdef __aarch64__
    __builtin_prefetch(ptr, 0, 3);
#else
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#endif
}

inline void prefetch_write(const void* ptr) {
#ifdef __aarch64__
    __builtin_prefetch(ptr, 1, 3);
#else
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#endif
}


class SIMDPortfolioOptimizer {
private:
    int num_assets_;
    aligned_vector<double> expected_returns_;
    aligned_vector<double> covariance_matrix_flat_;  
    aligned_vector<double> lower_bounds_;
    aligned_vector<double> upper_bounds_;
    
    
    double* get_covariance_element(int i, int j) const {
        return const_cast<double*>(&covariance_matrix_flat_[i * num_assets_ + j]);
    }
    
    
    void matrix_vector_multiply_simd_optimized(const double* matrix, const double* vector, 
                                             double* result, int n) const {
#ifdef __aarch64__
        
        for (int i = 0; i < n; i++) {
            const double* row = matrix + i * n;
            prefetch_read(row + 16);  
            
            float64x2_t sum1 = vdupq_n_f64(0.0);
            float64x2_t sum2 = vdupq_n_f64(0.0);
            
            int j = 0;
            
            for (; j <= n - 4; j += 4) {
                float64x2_t m1 = vld1q_f64(&row[j]);
                float64x2_t m2 = vld1q_f64(&row[j + 2]);
                float64x2_t v1 = vld1q_f64(&vector[j]);
                float64x2_t v2 = vld1q_f64(&vector[j + 2]);
                
                sum1 = vfmaq_f64(sum1, m1, v1);
                sum2 = vfmaq_f64(sum2, m2, v2);
            }
            
            
            float64x2_t total_sum = vaddq_f64(sum1, sum2);
            result[i] = vgetq_lane_f64(total_sum, 0) + vgetq_lane_f64(total_sum, 1);
            
            
            for (; j < n; j++) {
                result[i] += row[j] * vector[j];
            }
        }
#else
        
        for (int i = 0; i < n; i++) {
            const double* row = matrix + i * n;
            _mm_prefetch(reinterpret_cast<const char*>(row + 8), _MM_HINT_T0);
            
            __m256d sum1 = _mm256_setzero_pd();
            __m256d sum2 = _mm256_setzero_pd();
            
            int j = 0;
            
            for (; j <= n - 8; j += 8) {
                __m256d m1 = _mm256_loadu_pd(&row[j]);
                __m256d m2 = _mm256_loadu_pd(&row[j + 4]);
                __m256d v1 = _mm256_loadu_pd(&vector[j]);
                __m256d v2 = _mm256_loadu_pd(&vector[j + 4]);
                
                sum1 = _mm256_fmadd_pd(m1, v1, sum1);
                sum2 = _mm256_fmadd_pd(m2, v2, sum2);
            }
            
            
            __m256d total_sum = _mm256_add_pd(sum1, sum2);
            double temp[4];
            _mm256_storeu_pd(temp, total_sum);
            result[i] = temp[0] + temp[1] + temp[2] + temp[3];
            
            
            for (; j < n; j++) {
                result[i] += row[j] * vector[j];
            }
        }
#endif
    }
    
    
    double dot_product_simd_optimized(const double* a, const double* b, int n) const {
#ifdef __aarch64__
        float64x2_t sum1 = vdupq_n_f64(0.0);
        float64x2_t sum2 = vdupq_n_f64(0.0);
        float64x2_t sum3 = vdupq_n_f64(0.0);
        float64x2_t sum4 = vdupq_n_f64(0.0);
        
        int i = 0;
        
        for (; i <= n - 8; i += 8) {
            float64x2_t a1 = vld1q_f64(&a[i]);
            float64x2_t a2 = vld1q_f64(&a[i + 2]);
            float64x2_t a3 = vld1q_f64(&a[i + 4]);
            float64x2_t a4 = vld1q_f64(&a[i + 6]);
            
            float64x2_t b1 = vld1q_f64(&b[i]);
            float64x2_t b2 = vld1q_f64(&b[i + 2]);
            float64x2_t b3 = vld1q_f64(&b[i + 4]);
            float64x2_t b4 = vld1q_f64(&b[i + 6]);
            
            sum1 = vfmaq_f64(sum1, a1, b1);
            sum2 = vfmaq_f64(sum2, a2, b2);
            sum3 = vfmaq_f64(sum3, a3, b3);
            sum4 = vfmaq_f64(sum4, a4, b4);
        }
        
        
        float64x2_t total_sum = vaddq_f64(vaddq_f64(sum1, sum2), vaddq_f64(sum3, sum4));
        double result = vgetq_lane_f64(total_sum, 0) + vgetq_lane_f64(total_sum, 1);
        
        
        for (; i < n; i++) {
            result += a[i] * b[i];
        }
        
        return result;
#else
        __m256d sum1 = _mm256_setzero_pd();
        __m256d sum2 = _mm256_setzero_pd();
        __m256d sum3 = _mm256_setzero_pd();
        __m256d sum4 = _mm256_setzero_pd();
        
        int i = 0;
        
        for (; i <= n - 16; i += 16) {
            __m256d a1 = _mm256_loadu_pd(&a[i]);
            __m256d a2 = _mm256_loadu_pd(&a[i + 4]);
            __m256d a3 = _mm256_loadu_pd(&a[i + 8]);
            __m256d a4 = _mm256_loadu_pd(&a[i + 12]);
            
            __m256d b1 = _mm256_loadu_pd(&b[i]);
            __m256d b2 = _mm256_loadu_pd(&b[i + 4]);
            __m256d b3 = _mm256_loadu_pd(&b[i + 8]);
            __m256d b4 = _mm256_loadu_pd(&b[i + 12]);
            
            sum1 = _mm256_fmadd_pd(a1, b1, sum1);
            sum2 = _mm256_fmadd_pd(a2, b2, sum2);
            sum3 = _mm256_fmadd_pd(a3, b3, sum3);
            sum4 = _mm256_fmadd_pd(a4, b4, sum4);
        }
        
        
        __m256d total_sum = _mm256_add_pd(_mm256_add_pd(sum1, sum2), _mm256_add_pd(sum3, sum4));
        double temp[4];
        _mm256_storeu_pd(temp, total_sum);
        double result = temp[0] + temp[1] + temp[2] + temp[3];
        
        
        for (; i < n; i++) {
            result += a[i] * b[i];
        }
        
        return result;
#endif
    }
    
    
    void vector_operations_simd(const double* a, const double* b, double* result, 
                               int n, char operation) const {
#ifdef __aarch64__
        int i = 0;
        for (; i <= n - 2; i += 2) {
            prefetch_read(&a[i + 16]);
            prefetch_read(&b[i + 16]);
            prefetch_write(&result[i + 16]);
            
            float64x2_t a_vals = vld1q_f64(&a[i]);
            float64x2_t b_vals = vld1q_f64(&b[i]);
            float64x2_t res;
            
            switch (operation) {
                case '+': res = vaddq_f64(a_vals, b_vals); break;
                case '-': res = vsubq_f64(a_vals, b_vals); break;
                case '*': res = vmulq_f64(a_vals, b_vals); break;
                default: res = vaddq_f64(a_vals, b_vals); break;
            }
            
            vst1q_f64(&result[i], res);
        }
        
        for (; i < n; i++) {
            switch (operation) {
                case '+': result[i] = a[i] + b[i]; break;
                case '-': result[i] = a[i] - b[i]; break;
                case '*': result[i] = a[i] * b[i]; break;
                default: result[i] = a[i] + b[i]; break;
            }
        }
#else
        int i = 0;
        for (; i <= n - 4; i += 4) {
            _mm_prefetch(reinterpret_cast<const char*>(&a[i + 16]), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&b[i + 16]), _MM_HINT_T0);
            
            __m256d a_vals = _mm256_loadu_pd(&a[i]);
            __m256d b_vals = _mm256_loadu_pd(&b[i]);
            __m256d res;
            
            switch (operation) {
                case '+': res = _mm256_add_pd(a_vals, b_vals); break;
                case '-': res = _mm256_sub_pd(a_vals, b_vals); break;
                case '*': res = _mm256_mul_pd(a_vals, b_vals); break;
                default: res = _mm256_add_pd(a_vals, b_vals); break;
            }
            
            _mm256_storeu_pd(&result[i], res);
        }
        
        for (; i < n; i++) {
            switch (operation) {
                case '+': result[i] = a[i] + b[i]; break;
                case '-': result[i] = a[i] - b[i]; break;
                case '*': result[i] = a[i] * b[i]; break;
                default: result[i] = a[i] + b[i]; break;
            }
        }
#endif
    }
    
    
    std::vector<double> matrix_vector_multiply_simd(const std::vector<std::vector<double>>& matrix,
                                                    const std::vector<double>& vector) {
        int n = matrix.size();
        std::vector<double> result(n);
        aligned_vector<double> flat_matrix(n * n);
        
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                flat_matrix[i * n + j] = matrix[i][j];
            }
        }
        
        matrix_vector_multiply_simd_optimized(flat_matrix.data(), vector.data(), result.data(), n);
        return result;
    }
    
    double dot_product_simd(const std::vector<double>& a, const std::vector<double>& b) {
        return dot_product_simd_optimized(a.data(), b.data(), a.size());
    }
    
    void vector_add_simd(const std::vector<double>& a, const std::vector<double>& b, 
                         std::vector<double>& result) {
        result.resize(a.size());
        vector_operations_simd(a.data(), b.data(), result.data(), a.size(), '+');
    }
    
    void vector_scale_simd(const std::vector<double>& a, double scale, std::vector<double>& result) {
        result.resize(a.size());
        aligned_vector<double> scale_vec(a.size(), scale);
        vector_operations_simd(a.data(), scale_vec.data(), result.data(), a.size(), '*');
    }
    
public:
    SIMDPortfolioOptimizer(int num_assets) : num_assets_(num_assets) {
        expected_returns_.resize(num_assets);
        covariance_matrix_flat_.resize(num_assets * num_assets);
        lower_bounds_.resize(num_assets, 0.0);
        upper_bounds_.resize(num_assets, 1.0);
    }
    
    void set_expected_returns(const std::vector<double>& returns) {
        expected_returns_.assign(returns.begin(), returns.end());
    }
    
    void set_covariance_matrix(const std::vector<std::vector<double>>& covariance) {
        
        for (int i = 0; i < num_assets_; i++) {
            for (int j = 0; j < num_assets_; j++) {
                covariance_matrix_flat_[i * num_assets_ + j] = covariance[i][j];
            }
        }
    }
    
    void set_bounds(const std::vector<double>& lower, const std::vector<double>& upper) {
        lower_bounds_.assign(lower.begin(), lower.end());
        upper_bounds_.assign(upper.begin(), upper.end());
    }
    
    
    double calculate_portfolio_return(const std::vector<double>& weights) {
        return dot_product_simd_optimized(weights.data(), expected_returns_.data(), num_assets_);
    }
    
    double calculate_portfolio_return(const aligned_vector<double>& weights) {
        return dot_product_simd_optimized(weights.data(), expected_returns_.data(), num_assets_);
    }
    
    
    double calculate_portfolio_variance(const std::vector<double>& weights) {
        aligned_vector<double> cov_weights(num_assets_);
        matrix_vector_multiply_simd_optimized(covariance_matrix_flat_.data(), weights.data(), 
                                            cov_weights.data(), num_assets_);
        return dot_product_simd_optimized(weights.data(), cov_weights.data(), num_assets_);
    }
    
    double calculate_portfolio_variance(const aligned_vector<double>& weights) {
        aligned_vector<double> cov_weights(num_assets_);
        matrix_vector_multiply_simd_optimized(covariance_matrix_flat_.data(), weights.data(), 
                                            cov_weights.data(), num_assets_);
        return dot_product_simd_optimized(weights.data(), cov_weights.data(), num_assets_);
    }
    
    
    double calculate_sharpe_ratio(const std::vector<double>& weights, double risk_free_rate = 0.0) {
        double portfolio_return = calculate_portfolio_return(weights);
        double portfolio_variance = calculate_portfolio_variance(weights);
        
        if (portfolio_variance <= 1e-12 || !std::isfinite(portfolio_variance)) return 0.0;
        
        double excess_return = portfolio_return - risk_free_rate;
        double volatility = std::sqrt(portfolio_variance);
        
        if (volatility <= 1e-12 || !std::isfinite(volatility)) return 0.0;
        
        double sharpe = excess_return / volatility;
        return std::isfinite(sharpe) ? sharpe : 0.0;
    }
    
    
    std::vector<double> optimize_mean_variance_advanced(double target_return, int max_iterations = 2000) {
        aligned_vector<double> weights(num_assets_, 1.0 / num_assets_);
        aligned_vector<double> gradient(num_assets_);
        aligned_vector<double> momentum(num_assets_, 0.0);
        
        double learning_rate = 0.005;
        const double momentum_factor = 0.9;
        const double tolerance = 1e-8;
        const double regularization = 1e-10;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            
            aligned_vector<double> cov_weights(num_assets_);
            matrix_vector_multiply_simd_optimized(covariance_matrix_flat_.data(), weights.data(), 
                                                cov_weights.data(), num_assets_);
            
            
            for (int i = 0; i < num_assets_; i++) {
                gradient[i] = 2.0 * cov_weights[i] + regularization * weights[i];
            }
            
            
            double current_return = calculate_portfolio_return(weights);
            double return_error = current_return - target_return;
            
            for (int i = 0; i < num_assets_; i++) {
                gradient[i] += 2.0 * return_error * expected_returns_[i];
            }
            
            
            for (int i = 0; i < num_assets_; i++) {
                momentum[i] = momentum_factor * momentum[i] + learning_rate * gradient[i];
                weights[i] -= momentum[i];
            }
            
            
            apply_constraints_advanced(weights);
            
            
            double gradient_norm = std::sqrt(dot_product_simd_optimized(gradient.data(), gradient.data(), num_assets_));
            if (gradient_norm < tolerance) break;
            
            
            if (iter % 200 == 0) {
                learning_rate *= 0.98;
            }
        }
        
        return std::vector<double>(weights.begin(), weights.end());
    }
    
    
    std::vector<double> optimize_max_sharpe_advanced(double risk_free_rate = 0.0, int max_iterations = 2000) {
        aligned_vector<double> weights(num_assets_, 1.0 / num_assets_);
        aligned_vector<double> gradient(num_assets_);
        aligned_vector<double> momentum(num_assets_, 0.0);
        
        double learning_rate = 0.01;
        const double momentum_factor = 0.9;
        const double tolerance = 1e-8;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            double portfolio_return = calculate_portfolio_return(weights);
            double portfolio_variance = calculate_portfolio_variance(weights);
            double portfolio_volatility = std::sqrt(portfolio_variance + 1e-12);
            
            if (portfolio_volatility <= 0) break;
            
            double excess_return = portfolio_return - risk_free_rate;
            
            
            aligned_vector<double> cov_weights(num_assets_);
            matrix_vector_multiply_simd_optimized(covariance_matrix_flat_.data(), weights.data(), 
                                                cov_weights.data(), num_assets_);
            
            for (int i = 0; i < num_assets_; i++) {
                gradient[i] = (expected_returns_[i] - risk_free_rate) / portfolio_volatility - 
                             excess_return * cov_weights[i] / (portfolio_volatility * portfolio_variance);
            }
            
            
            for (int i = 0; i < num_assets_; i++) {
                momentum[i] = momentum_factor * momentum[i] + learning_rate * gradient[i];
                weights[i] += momentum[i];
            }
            
            
            apply_constraints_advanced(weights);
            
            
            double gradient_norm = std::sqrt(dot_product_simd_optimized(gradient.data(), gradient.data(), num_assets_));
            if (gradient_norm < tolerance) break;
            
            
            if (iter % 200 == 0) {
                learning_rate *= 0.98;
            }
        }
        
        return std::vector<double>(weights.begin(), weights.end());
    }
    
    
    std::vector<double> optimize_risk_parity_advanced(int max_iterations = 2000) {
        aligned_vector<double> weights(num_assets_, 1.0 / num_assets_);
        aligned_vector<double> gradient(num_assets_);
        aligned_vector<double> momentum(num_assets_, 0.0);
        
        double learning_rate = 0.01;
        const double momentum_factor = 0.9;
        const double tolerance = 1e-8;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            double portfolio_variance = calculate_portfolio_variance(weights);
            
            if (portfolio_variance <= 1e-12) break;
            
            aligned_vector<double> cov_weights(num_assets_);
            matrix_vector_multiply_simd_optimized(covariance_matrix_flat_.data(), weights.data(), 
                                                cov_weights.data(), num_assets_);
            
            
            aligned_vector<double> marginal_risk(num_assets_);
            for (int i = 0; i < num_assets_; i++) {
                marginal_risk[i] = cov_weights[i] / std::sqrt(portfolio_variance);
            }
            
            
            double target_risk_contrib = std::sqrt(portfolio_variance) / num_assets_;
            
            
            for (int i = 0; i < num_assets_; i++) {
                double current_risk_contrib = weights[i] * marginal_risk[i];
                gradient[i] = 2.0 * (current_risk_contrib - target_risk_contrib) * marginal_risk[i];
            }
            
            
            for (int i = 0; i < num_assets_; i++) {
                momentum[i] = momentum_factor * momentum[i] + learning_rate * gradient[i];
                weights[i] -= momentum[i];
            }
            
            
            apply_constraints_advanced(weights);
            
            
            double gradient_norm = std::sqrt(dot_product_simd_optimized(gradient.data(), gradient.data(), num_assets_));
            if (gradient_norm < tolerance) break;
            
            
            if (iter % 200 == 0) {
                learning_rate *= 0.98;
            }
        }        
        return std::vector<double>(weights.begin(), weights.end());
    }
    
    
    std::vector<double> optimize_max_sharpe(double risk_free_rate = 0.0, int max_iterations = 2000) {
        return optimize_max_sharpe_advanced(risk_free_rate, max_iterations);
    }
    
    std::vector<double> optimize_min_variance(int max_iterations = 2000) {
        aligned_vector<double> weights(num_assets_, 1.0 / num_assets_);
        aligned_vector<double> gradient(num_assets_);
        aligned_vector<double> momentum(num_assets_, 0.0);
        
        double learning_rate = 0.01;
        const double momentum_factor = 0.9;
        const double tolerance = 1e-8;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            
            matrix_vector_multiply_simd_optimized(covariance_matrix_flat_.data(), weights.data(), 
                                                gradient.data(), num_assets_);
            
            
            for (int i = 0; i < num_assets_; i++) {
                gradient[i] *= 2.0;
            }
            
            
            for (int i = 0; i < num_assets_; i++) {
                momentum[i] = momentum_factor * momentum[i] + learning_rate * gradient[i];
                weights[i] -= momentum[i];
            }
            
            
            apply_constraints_advanced(weights);
            
            
            double gradient_norm = std::sqrt(dot_product_simd_optimized(gradient.data(), gradient.data(), num_assets_));
            if (gradient_norm < tolerance) break;
            
            
            if (iter % 200 == 0) {
                learning_rate *= 0.98;
            }
        }
        
        return std::vector<double>(weights.begin(), weights.end());
    }
    
    std::vector<double> optimize_risk_parity(int max_iterations = 2000) {
        return optimize_risk_parity_advanced(max_iterations);
    }

private:
    void apply_constraints_advanced(aligned_vector<double>& weights) {
        
        for (int i = 0; i < num_assets_; i++) {
            weights[i] = std::max(lower_bounds_[i], std::min(upper_bounds_[i], weights[i]));
        }
        
        
        double sum = 0.0;
        for (int i = 0; i < num_assets_; i++) {
            sum += weights[i];
        }
        if (sum > 0) {
            for (int i = 0; i < num_assets_; i++) {
                weights[i] /= sum;
            }
        }
    }
    void apply_constraints(std::vector<double>& weights) {
        
        for (int i = 0; i < num_assets_; i++) {
            weights[i] = std::max(lower_bounds_[i], std::min(upper_bounds_[i], weights[i]));
        }
        
        
        double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (sum > 0) {
            for (double& w : weights) {
                w /= sum;
            }
        }
    }
};


class NaivePortfolioOptimizer {
private:
    int num_assets_;
    std::vector<double> expected_returns_;
    std::vector<std::vector<double>> covariance_matrix_;
    
    double dot_product_naive(const std::vector<double>& a, const std::vector<double>& b) {
        double result = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    std::vector<double> matrix_vector_multiply_naive(const std::vector<std::vector<double>>& matrix,
                                                    const std::vector<double>& vector) {
        int n = matrix.size();
        std::vector<double> result(n, 0.0);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        
        return result;
    }
    
public:
    NaivePortfolioOptimizer(int num_assets) : num_assets_(num_assets) {
        expected_returns_.resize(num_assets);
        covariance_matrix_.resize(num_assets, std::vector<double>(num_assets));
    }
    
    void set_expected_returns(const std::vector<double>& returns) {
        expected_returns_ = returns;
    }
    
    void set_covariance_matrix(const std::vector<std::vector<double>>& covariance) {
        covariance_matrix_ = covariance;
    }
    
    double calculate_portfolio_return(const std::vector<double>& weights) {
        return dot_product_naive(weights, expected_returns_);
    }
    
    double calculate_portfolio_variance(const std::vector<double>& weights) {
        auto cov_weights = matrix_vector_multiply_naive(covariance_matrix_, weights);
        return dot_product_naive(weights, cov_weights);
    }
    
    double calculate_sharpe_ratio(const std::vector<double>& weights, double risk_free_rate = 0.0) {
        double portfolio_return = calculate_portfolio_return(weights);
        double portfolio_variance = calculate_portfolio_variance(weights);
        
        if (portfolio_variance <= 1e-12 || !std::isfinite(portfolio_variance)) return 0.0;
        
        double excess_return = portfolio_return - risk_free_rate;
        double volatility = std::sqrt(portfolio_variance);
        
        if (volatility <= 1e-12 || !std::isfinite(volatility)) return 0.0;
        
        double sharpe = excess_return / volatility;
        return std::isfinite(sharpe) ? sharpe : 0.0;
    }
    
    std::vector<double> optimize_max_sharpe(double risk_free_rate = 0.0, int max_iterations = 2000) {
        std::vector<double> weights(num_assets_, 1.0 / num_assets_);
        
        double learning_rate = 0.01;
        const double tolerance = 1e-8;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            double portfolio_return = calculate_portfolio_return(weights);
            double portfolio_variance = calculate_portfolio_variance(weights);
            double portfolio_volatility = std::sqrt(portfolio_variance + 1e-12);
            
            if (portfolio_volatility <= 0) break;
            
            double excess_return = portfolio_return - risk_free_rate;
            
            auto cov_weights = matrix_vector_multiply_naive(covariance_matrix_, weights);
            
            std::vector<double> gradient(num_assets_);
            for (int i = 0; i < num_assets_; i++) {
                gradient[i] = (expected_returns_[i] - risk_free_rate) / portfolio_volatility - 
                             excess_return * cov_weights[i] / (portfolio_volatility * portfolio_variance);
            }
            
            
            for (int i = 0; i < num_assets_; i++) {
                weights[i] += learning_rate * gradient[i];
            }
            
            
            apply_constraints_naive(weights);
            
            
            double gradient_norm = 0.0;
            for (double g : gradient) {
                gradient_norm += g * g;
            }
            gradient_norm = std::sqrt(gradient_norm);
            if (gradient_norm < tolerance) break;
            
            
            if (iter % 200 == 0) {
                learning_rate *= 0.98;
            }
        }
        
        return weights;
    }
    
private:
    void apply_constraints_naive(std::vector<double>& weights) {
        
        double sum = 0.0;
        for (double w : weights) {
            sum += w;
        }
        if (sum > 0) {
            for (double& w : weights) {
                w /= sum;
            }
        }
    }
};


class PortfolioBenchmark {
private:
    std::vector<int> test_sizes_ = {10, 25, 50, 100, 200, 500};
    
    std::pair<std::vector<double>, std::vector<std::vector<double>>> 
    generate_test_data(int num_assets, int seed = 42) {
        std::mt19937 gen(seed);
        std::normal_distribution<> return_dist(0.08, 0.05);
        std::uniform_real_distribution<> vol_dist(0.10, 0.40);
        std::uniform_real_distribution<> corr_dist(0.1, 0.7);
        
        std::vector<double> expected_returns(num_assets);
        for (int i = 0; i < num_assets; i++) {
            expected_returns[i] = return_dist(gen);
        }
        
        std::vector<double> volatilities(num_assets);
        for (int i = 0; i < num_assets; i++) {
            volatilities[i] = vol_dist(gen);
        }
        
        
        std::vector<std::vector<double>> covariance(num_assets, std::vector<double>(num_assets));
        
        
        std::vector<std::vector<double>> A(num_assets, std::vector<double>(num_assets));
        std::uniform_real_distribution<> rand_dist(-1.0, 1.0);
        
        for (int i = 0; i < num_assets; i++) {
            for (int j = 0; j < num_assets; j++) {
                A[i][j] = rand_dist(gen);
            }
        }
        
        
        double lambda = 0.01; 
        for (int i = 0; i < num_assets; i++) {
            for (int j = 0; j < num_assets; j++) {
                double sum = 0.0;
                for (int k = 0; k < num_assets; k++) {
                    sum += A[k][i] * A[k][j];
                }
                covariance[i][j] = sum * volatilities[i] * volatilities[j] / num_assets;
                if (i == j) {
                    covariance[i][j] += lambda; 
                }
            }
        }
        
        return {expected_returns, covariance};
    }
    
    template<typename Func>
    double measure_time(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start).count();
    }
    
public:
    void run_comprehensive_benchmark() {
        std::cout << "\n=== SIMD Portfolio Optimization Benchmark Suite ===\n";
        
        std::cout << std::left << std::setw(12) << "Assets" 
                  << std::setw(15) << "SIMD (μs)" 
                  << std::setw(15) << "Naive (μs)" 
                  << std::setw(12) << "Speedup" 
                  << std::setw(18) << "SIMD Sharpe" 
                  << std::setw(18) << "Naive Sharpe" 
                  << std::setw(15) << "Accuracy" << std::endl;
        std::cout << std::string(105, '-') << std::endl;
        
        for (int num_assets : test_sizes_) {
            run_size_benchmark(num_assets);
        }
        
        std::cout << "\n=== Matrix Operations Micro-benchmarks ===\n";
        run_matrix_operations_benchmark();
        
        std::cout << "\n=== Memory Access Pattern Analysis ===\n";
        run_memory_pattern_benchmark();
    }
    
private:
    void run_size_benchmark(int num_assets) {
        auto [expected_returns, covariance] = generate_test_data(num_assets);
        
        
        SIMDPortfolioOptimizer simd_optimizer(num_assets);
        NaivePortfolioOptimizer naive_optimizer(num_assets);
        
        simd_optimizer.set_expected_returns(expected_returns);
        simd_optimizer.set_covariance_matrix(covariance);
        naive_optimizer.set_expected_returns(expected_returns);
        naive_optimizer.set_covariance_matrix(covariance);
        
        
        std::vector<double> simd_weights;
        double simd_time = measure_time([&]() {
            simd_weights = simd_optimizer.optimize_max_sharpe(0.02, 500);
        });
        
        
        std::vector<double> naive_weights;
        double naive_time = measure_time([&]() {
            naive_weights = naive_optimizer.optimize_max_sharpe(0.02, 500);
        });
        
        
        double simd_sharpe = simd_optimizer.calculate_sharpe_ratio(simd_weights, 0.02);
        double naive_sharpe = naive_optimizer.calculate_sharpe_ratio(naive_weights, 0.02);
        double speedup = (simd_time > 0) ? naive_time / simd_time : 0.0;
        double accuracy = 0.0;
        if (std::abs(naive_sharpe) > 1e-12 && std::isfinite(naive_sharpe) && std::isfinite(simd_sharpe)) {
            accuracy = std::abs(simd_sharpe - naive_sharpe) / std::abs(naive_sharpe) * 100;
        }
        
        
        std::cout << std::left << std::setw(12) << num_assets
                  << std::setw(15) << std::fixed << std::setprecision(1) << simd_time
                  << std::setw(15) << std::fixed << std::setprecision(1) << naive_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x"
                  << std::setw(18) << std::fixed << std::setprecision(4) << (std::isfinite(simd_sharpe) ? simd_sharpe : 0.0)
                  << std::setw(18) << std::fixed << std::setprecision(4) << (std::isfinite(naive_sharpe) ? naive_sharpe : 0.0)
                  << std::setw(15) << std::fixed << std::setprecision(3) << accuracy << "%"
                  << std::endl;
    }
    
    void run_matrix_operations_benchmark() {
        std::cout << "\nOperation          | Size | SIMD (μs) | Naive (μs) | Speedup\n";
        std::cout << std::string(60, '-') << std::endl;
        
        for (int size : {100, 200, 500}) {
            benchmark_matrix_vector_multiply(size);
            benchmark_dot_product(size);
        }
    }
    
    void benchmark_matrix_vector_multiply(int size) {
        auto [expected_returns, covariance] = generate_test_data(size);
        std::vector<double> weights(size, 1.0 / size);
        
        SIMDPortfolioOptimizer simd_optimizer(size);
        NaivePortfolioOptimizer naive_optimizer(size);
        
        simd_optimizer.set_covariance_matrix(covariance);
        naive_optimizer.set_covariance_matrix(covariance);
        
        double simd_time = measure_time([&]() {
            for (int i = 0; i < 1000; i++) {
                simd_optimizer.calculate_portfolio_variance(weights);
            }
        });
        
        double naive_time = measure_time([&]() {
            for (int i = 0; i < 1000; i++) {
                naive_optimizer.calculate_portfolio_variance(weights);
            }
        });
        
        double speedup = (simd_time > 0) ? naive_time / simd_time : 0.0;
        
        std::cout << std::left << std::setw(19) << "Matrix*Vector"
                  << std::setw(6) << size
                  << std::setw(11) << std::fixed << std::setprecision(1) << simd_time
                  << std::setw(12) << std::fixed << std::setprecision(1) << naive_time
                  << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    
    void benchmark_dot_product(int size) {
        std::vector<double> a(size), b(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (int i = 0; i < size; i++) {
            a[i] = dis(gen);
            b[i] = dis(gen);
        }
        
        SIMDPortfolioOptimizer simd_optimizer(size);
        NaivePortfolioOptimizer naive_optimizer(size);
        
        simd_optimizer.set_expected_returns(a);
        naive_optimizer.set_expected_returns(a);
        
        double simd_time = measure_time([&]() {
            for (int i = 0; i < 10000; i++) {
                simd_optimizer.calculate_portfolio_return(b);
            }
        });
        
        double naive_time = measure_time([&]() {
            for (int i = 0; i < 10000; i++) {
                naive_optimizer.calculate_portfolio_return(b);
            }
        });
        
        double speedup = (simd_time > 0) ? naive_time / simd_time : 0.0;
        
        std::cout << std::left << std::setw(19) << "Dot Product"
                  << std::setw(6) << size
                  << std::setw(11) << std::fixed << std::setprecision(1) << simd_time
                  << std::setw(12) << std::fixed << std::setprecision(1) << naive_time
                  << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    
    void run_memory_pattern_benchmark() {
        std::cout << "\nMemory Access Pattern Analysis:\n";
        std::cout << "- Aligned vs unaligned memory access\n";
        std::cout << "- Cache-friendly vs cache-unfriendly patterns\n\n";
        
        benchmark_memory_alignment();
        benchmark_cache_patterns();
    }
    
    void benchmark_memory_alignment() {
        const int size = 1000;
        const int iterations = 10000;
        
        
        aligned_vector<double> aligned_data(size);
        std::vector<double> unaligned_data(size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (int i = 0; i < size; i++) {
            aligned_data[i] = dis(gen);
            unaligned_data[i] = dis(gen);
        }
        
        double aligned_time = measure_time([&]() {
            volatile double sum = 0.0;
            for (int iter = 0; iter < iterations; iter++) {
                for (int i = 0; i < size; i++) {
                    sum += aligned_data[i];
                }
            }
        });
        
        double unaligned_time = measure_time([&]() {
            volatile double sum = 0.0;
            for (int iter = 0; iter < iterations; iter++) {
                for (int i = 0; i < size; i++) {
                    sum += unaligned_data[i];
                }
            }
        });
        
        double speedup = (aligned_time > 0) ? unaligned_time / aligned_time : 0.0;
        
        std::cout << "Memory Alignment Impact:\n";
        std::cout << "- Aligned:   " << std::fixed << std::setprecision(1) << aligned_time << " μs\n";
        std::cout << "- Unaligned: " << std::fixed << std::setprecision(1) << unaligned_time << " μs\n";
        std::cout << "- Speedup:   " << std::fixed << std::setprecision(2) << speedup << "x\n\n";
    }
    
    void benchmark_cache_patterns() {
        const int size = 500;
        const int iterations = 100;
        
        std::vector<std::vector<double>> matrix(size, std::vector<double>(size));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrix[i][j] = dis(gen);
            }
        }
        
        
        double row_major_time = measure_time([&]() {
            volatile double sum = 0.0;
            for (int iter = 0; iter < iterations; iter++) {
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        sum += matrix[i][j];
                    }
                }
            }
        });
        
        
        double col_major_time = measure_time([&]() {
            volatile double sum = 0.0;
            for (int iter = 0; iter < iterations; iter++) {
                for (int j = 0; j < size; j++) {
                    for (int i = 0; i < size; i++) {
                        sum += matrix[i][j];
                    }
                }
            }
        });
        
        double speedup = (row_major_time > 0) ? col_major_time / row_major_time : 0.0;
        
        std::cout << "Cache Access Pattern Impact:\n";
        std::cout << "- Row-major:    " << std::fixed << std::setprecision(1) << row_major_time << " μs\n";
        std::cout << "- Column-major: " << std::fixed << std::setprecision(1) << col_major_time << " μs\n";
        std::cout << "- Speedup:      " << std::fixed << std::setprecision(2) << speedup << "x\n\n";
    }
};


void benchmark_optimization() {
    std::cout << "SIMD Portfolio Optimization Legacy Benchmark\n";
    
    std::vector<int> num_assets_list = {10, 50, 100, 200};
    
    for (int num_assets : num_assets_list) {
        std::cout << "Portfolio with " << num_assets << " assets:\n";
        
        SIMDPortfolioOptimizer optimizer(num_assets);
        
        
        std::vector<double> expected_returns(num_assets);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> return_dist(0.08, 0.05);
        
        for (int i = 0; i < num_assets; i++) {
            expected_returns[i] = return_dist(gen);
        }
        
        
        std::vector<std::vector<double>> covariance(num_assets, std::vector<double>(num_assets));
        std::uniform_real_distribution<> corr_dist(0.1, 0.8);
        std::uniform_real_distribution<> vol_dist(0.15, 0.35);
        
        std::vector<double> volatilities(num_assets);
        for (int i = 0; i < num_assets; i++) {
            volatilities[i] = vol_dist(gen);
        }
        
        
        std::vector<std::vector<double>> temp_matrix(num_assets, std::vector<double>(num_assets));
        for (int i = 0; i < num_assets; i++) {
            for (int j = 0; j < num_assets; j++) {
                temp_matrix[i][j] = corr_dist(gen);
            }
        }
        
        
        for (int i = 0; i < num_assets; i++) {
            for (int j = 0; j < num_assets; j++) {
                if (i == j) {
                    covariance[i][j] = volatilities[i] * volatilities[i];
                } else {
                    double correlation = std::min(0.8, std::max(-0.8, temp_matrix[i][j]));
                    covariance[i][j] = correlation * volatilities[i] * volatilities[j];
                }
            }
        }
        
        
        for (int i = 0; i < num_assets; i++) {
            covariance[i][i] += 1e-6;
        }
        
        optimizer.set_expected_returns(expected_returns);
        optimizer.set_covariance_matrix(covariance);
        
        
        auto start = std::chrono::high_resolution_clock::now();
        auto max_sharpe_weights = optimizer.optimize_max_sharpe(0.02);
        auto end = std::chrono::high_resolution_clock::now();
        auto max_sharpe_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto min_var_weights = optimizer.optimize_min_variance();
        end = std::chrono::high_resolution_clock::now();
        auto min_var_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        auto risk_parity_weights = optimizer.optimize_risk_parity();
        end = std::chrono::high_resolution_clock::now();
        auto risk_parity_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        
        double max_sharpe_return = optimizer.calculate_portfolio_return(max_sharpe_weights);
        double max_sharpe_variance = optimizer.calculate_portfolio_variance(max_sharpe_weights);
        double max_sharpe_ratio = optimizer.calculate_sharpe_ratio(max_sharpe_weights, 0.02);
        
        double min_var_return = optimizer.calculate_portfolio_return(min_var_weights);
        double min_var_variance = optimizer.calculate_portfolio_variance(min_var_weights);
        
        double rp_return = optimizer.calculate_portfolio_return(risk_parity_weights);
        double rp_variance = optimizer.calculate_portfolio_variance(risk_parity_weights);
        
        std::cout << "  Max Sharpe optimization: " << max_sharpe_time.count() << " μs\n";
        std::cout << "    Return: " << std::fixed << std::setprecision(4) << max_sharpe_return << "\n";
        std::cout << "    Volatility: " << std::fixed << std::setprecision(4) << std::sqrt(max_sharpe_variance) << "\n";
        std::cout << "    Sharpe Ratio: " << std::fixed << std::setprecision(4) << max_sharpe_ratio << "\n";
        
        std::cout << "  Min Variance optimization: " << min_var_time.count() << " μs\n";
        std::cout << "    Return: " << std::fixed << std::setprecision(4) << min_var_return << "\n";
        std::cout << "    Volatility: " << std::fixed << std::setprecision(4) << std::sqrt(min_var_variance) << "\n";
        
        std::cout << "  Risk Parity optimization: " << risk_parity_time.count() << " μs\n";
        std::cout << "    Return: " << std::fixed << std::setprecision(4) << rp_return << "\n";
        std::cout << "    Volatility: " << std::fixed << std::setprecision(4) << std::sqrt(rp_variance) << "\n\n";
    }
}

int main() {
    std::cout << "=== SIMD-Optimized Portfolio Optimization Engine ===\n\n";
    
    
    PortfolioBenchmark benchmark;
    benchmark.run_comprehensive_benchmark();
    
    
    std::cout << "\n\n";
    benchmark_optimization();
    
    std::cout << "\n=== Portfolio Analysis Example ===\n";
    
    
    SIMDPortfolioOptimizer optimizer(5);
    
    std::vector<double> expected_returns = {0.08, 0.12, 0.10, 0.15, 0.06};
    std::vector<std::vector<double>> covariance = {
        {0.04, 0.01, 0.02, 0.015, 0.005},
        {0.01, 0.09, 0.03, 0.02, 0.01},
        {0.02, 0.03, 0.16, 0.04, 0.015},
        {0.015, 0.02, 0.04, 0.25, 0.02},
        {0.005, 0.01, 0.015, 0.02, 0.01}
    };
    
    optimizer.set_expected_returns(expected_returns);
    optimizer.set_covariance_matrix(covariance);
    
    
    auto max_sharpe_weights = optimizer.optimize_max_sharpe(0.03);
    auto min_var_weights = optimizer.optimize_min_variance();
    auto risk_parity_weights = optimizer.optimize_risk_parity();
    
    std::cout << "\nOptimal Portfolio Weights:\n";
    std::cout << "Asset\tMax Sharpe\tMin Var\tRisk Parity\n\n";
    
    for (int i = 0; i < 5; i++) {
        std::cout << i + 1 << "\t" 
                  << std::fixed << std::setprecision(4) << max_sharpe_weights[i] << "\t\t"
                  << std::fixed << std::setprecision(4) << min_var_weights[i] << "\t"
                  << std::fixed << std::setprecision(4) << risk_parity_weights[i] << "\n";
    }
    
    
    std::cout << "\nPerformance Metrics:\n";
    std::cout << "Strategy\t\tReturn\t\tVolatility\tSharpe Ratio\n\n";
    
    double ms_return = optimizer.calculate_portfolio_return(max_sharpe_weights);
    double ms_variance = optimizer.calculate_portfolio_variance(max_sharpe_weights);
    double ms_sharpe = optimizer.calculate_sharpe_ratio(max_sharpe_weights, 0.03);
    
    double mv_return = optimizer.calculate_portfolio_return(min_var_weights);
    double mv_variance = optimizer.calculate_portfolio_variance(min_var_weights);
    double mv_sharpe = optimizer.calculate_sharpe_ratio(min_var_weights, 0.03);
    
    double rp_return = optimizer.calculate_portfolio_return(risk_parity_weights);
    double rp_variance = optimizer.calculate_portfolio_variance(risk_parity_weights);
    double rp_sharpe = optimizer.calculate_sharpe_ratio(risk_parity_weights, 0.03);
    
    std::cout << "Max Sharpe\t\t" << std::fixed << std::setprecision(4) << ms_return 
              << "\t\t" << std::fixed << std::setprecision(4) << std::sqrt(ms_variance)
              << "\t\t" << std::fixed << std::setprecision(4) << ms_sharpe << "\n";
    
    std::cout << "Min Variance\t\t" << std::fixed << std::setprecision(4) << mv_return 
              << "\t\t" << std::fixed << std::setprecision(4) << std::sqrt(mv_variance)
              << "\t\t" << std::fixed << std::setprecision(4) << mv_sharpe << "\n";
    
    std::cout << "Risk Parity\t\t" << std::fixed << std::setprecision(4) << rp_return 
              << "\t\t" << std::fixed << std::setprecision(4) << std::sqrt(rp_variance)
              << "\t\t" << std::fixed << std::setprecision(4) << rp_sharpe << "\n";

    return 0;
}
