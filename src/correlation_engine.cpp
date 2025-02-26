#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <memory>
#include <algorithm>

#ifdef __aarch64__
    #include <arm_neon.h>
#else
    #include <immintrin.h>
#endif

class FinancialCorrelationEngine {
private:
    int num_assets;
    int num_observations;
    
    std::vector<double> aligned_buffer;
    std::vector<double> means_buffer;
    std::vector<double> variance_buffer;
    
public:
    FinancialCorrelationEngine(int assets, int observations) 
        : num_assets(assets), num_observations(observations) {
        aligned_buffer.reserve(num_assets * num_observations);
        means_buffer.resize(num_assets);
        variance_buffer.resize(num_assets);
    }
    
    std::vector<std::vector<double>> calculate_correlation_regular(const std::vector<std::vector<double>>& returns) {
        std::vector<std::vector<double>> correlation(num_assets, std::vector<double>(num_assets, 0.0));
        
        for (int i = 0; i < num_assets; i++) {
            double sum = 0.0;
            for (int j = 0; j < num_observations; j++) {
                sum += returns[i][j];
            }
            means_buffer[i] = sum / num_observations;
        }
        
        for (int i = 0; i < num_assets; i++) {
            for (int j = i; j < num_assets; j++) {
                double numerator = 0.0;
                double sum_sq_i = 0.0;
                double sum_sq_j = 0.0;
                
                for (int k = 0; k < num_observations; k++) {
                    double diff_i = returns[i][k] - means_buffer[i];
                    double diff_j = returns[j][k] - means_buffer[j];
                    
                    numerator += diff_i * diff_j;
                    sum_sq_i += diff_i * diff_i;
                    sum_sq_j += diff_j * diff_j;
                }
                
                double correlation_coeff = numerator / (std::sqrt(sum_sq_i * sum_sq_j) + 1e-12);
                correlation[i][j] = correlation_coeff;
                correlation[j][i] = correlation_coeff;
            }
        }
        
        return correlation;
    }
    
    std::vector<std::vector<double>> calculate_correlation_simd(const std::vector<std::vector<double>>& returns) {
        std::vector<std::vector<double>> correlation(num_assets, std::vector<double>(num_assets, 0.0));
        
        calculate_means_simd(returns);
        
        for (int i = 0; i < num_assets; i++) {
            correlation[i][i] = 1.0;
            
            for (int j = i + 1; j < num_assets; j++) {
                double correlation_coeff = calculate_pairwise_correlation_simd(returns[i], returns[j], 
                                                                             means_buffer[i], means_buffer[j]);
                correlation[i][j] = correlation_coeff;
                correlation[j][i] = correlation_coeff;
            }
        }
        
        return correlation;
    }
    
    
public:
    void calculate_means_simd(const std::vector<std::vector<double>>& returns) {
        for (int i = 0; i < num_assets; i++) {
            double sum = 0.0;
            
#ifdef __aarch64__
            // ARM NEON version
            const int simd_size = num_observations / 2;
            float64x2_t vec_sum = vdupq_n_f64(0.0);
            
            for (int k = 0; k < simd_size; k++) {
                float64x2_t vec_data = vld1q_f64(&returns[i][k * 2]);
                vec_sum = vaddq_f64(vec_sum, vec_data);
            }
            
            sum = vgetq_lane_f64(vec_sum, 0) + vgetq_lane_f64(vec_sum, 1);
            
            for (int k = simd_size * 2; k < num_observations; k++) {
                sum += returns[i][k];
            }
#else
            // AVX version
            const int simd_size = num_observations / 4;
            __m256d vec_sum = _mm256_setzero_pd();
            
            for (int k = 0; k < simd_size; k++) {
                __m256d vec_data = _mm256_loadu_pd(&returns[i][k * 4]);
                vec_sum = _mm256_add_pd(vec_sum, vec_data);
            }
            
            double temp[4];
            _mm256_storeu_pd(temp, vec_sum);
            sum = temp[0] + temp[1] + temp[2] + temp[3];
            
            for (int k = simd_size * 4; k < num_observations; k++) {
                sum += returns[i][k];
            }
#endif
            
            means_buffer[i] = sum / num_observations;
        }
    }
    
    double calculate_pairwise_correlation_simd(const std::vector<double>& series1, 
                                              const std::vector<double>& series2,
                                              double mean1, double mean2) {
        double numerator = 0.0;
        double sum_sq_1 = 0.0;
        double sum_sq_2 = 0.0;
        
#ifdef __aarch64__
        // ARM NEON version 
        const int simd_size = num_observations / 2;
        
        if (simd_size > 0) {
            const float64x2_t vec_mean1 = vdupq_n_f64(mean1);
            const float64x2_t vec_mean2 = vdupq_n_f64(mean2);
            
            float64x2_t vec_numerator = vdupq_n_f64(0.0);
            float64x2_t vec_sum_sq_1 = vdupq_n_f64(0.0);
            float64x2_t vec_sum_sq_2 = vdupq_n_f64(0.0);
            
            for (int k = 0; k < simd_size; k++) {
                const float64x2_t vec_s1 = vld1q_f64(&series1[k * 2]);
                const float64x2_t vec_s2 = vld1q_f64(&series2[k * 2]);
                
                const float64x2_t diff1 = vsubq_f64(vec_s1, vec_mean1);
                const float64x2_t diff2 = vsubq_f64(vec_s2, vec_mean2);
                
                vec_numerator = vfmaq_f64(vec_numerator, diff1, diff2);
                vec_sum_sq_1 = vfmaq_f64(vec_sum_sq_1, diff1, diff1);
                vec_sum_sq_2 = vfmaq_f64(vec_sum_sq_2, diff2, diff2);
            }
            
            numerator = vgetq_lane_f64(vec_numerator, 0) + vgetq_lane_f64(vec_numerator, 1);
            sum_sq_1 = vgetq_lane_f64(vec_sum_sq_1, 0) + vgetq_lane_f64(vec_sum_sq_1, 1);
            sum_sq_2 = vgetq_lane_f64(vec_sum_sq_2, 0) + vgetq_lane_f64(vec_sum_sq_2, 1);
        }
        
        for (int k = simd_size * 2; k < num_observations; k++) {
            const double diff1 = series1[k] - mean1;
            const double diff2 = series2[k] - mean2;
            
            numerator += diff1 * diff2;
            sum_sq_1 += diff1 * diff1;
            sum_sq_2 += diff2 * diff2;
        }
#else
        // AVX version 
        const int simd_size = num_observations / 4;
        
        if (simd_size > 0) {
            const __m256d vec_mean1 = _mm256_set1_pd(mean1);
            const __m256d vec_mean2 = _mm256_set1_pd(mean2);
            
            __m256d vec_numerator = _mm256_setzero_pd();
            __m256d vec_sum_sq_1 = _mm256_setzero_pd();
            __m256d vec_sum_sq_2 = _mm256_setzero_pd();
            
            for (int k = 0; k < simd_size; k++) {
                const __m256d vec_s1 = _mm256_loadu_pd(&series1[k * 4]);
                const __m256d vec_s2 = _mm256_loadu_pd(&series2[k * 4]);
                
                const __m256d diff1 = _mm256_sub_pd(vec_s1, vec_mean1);
                const __m256d diff2 = _mm256_sub_pd(vec_s2, vec_mean2);
                
                vec_numerator = _mm256_fmadd_pd(diff1, diff2, vec_numerator);
                vec_sum_sq_1 = _mm256_fmadd_pd(diff1, diff1, vec_sum_sq_1);
                vec_sum_sq_2 = _mm256_fmadd_pd(diff2, diff2, vec_sum_sq_2);
            }
            
            alignas(32) double temp[4];
            _mm256_store_pd(temp, vec_numerator);
            numerator = temp[0] + temp[1] + temp[2] + temp[3];
            
            _mm256_store_pd(temp, vec_sum_sq_1);
            sum_sq_1 = temp[0] + temp[1] + temp[2] + temp[3];
            
            _mm256_store_pd(temp, vec_sum_sq_2);
            sum_sq_2 = temp[0] + temp[1] + temp[2] + temp[3];
        }
        
        for (int k = simd_size * 4; k < num_observations; k++) {
            const double diff1 = series1[k] - mean1;
            const double diff2 = series2[k] - mean2;
            
            numerator += diff1 * diff2;
            sum_sq_1 += diff1 * diff1;
            sum_sq_2 += diff2 * diff2;
        }
#endif
        
        return numerator / (std::sqrt(sum_sq_1 * sum_sq_2) + 1e-12);
    }
    
    std::pair<double, double> calculate_pairwise_correlation_and_covariance_simd(
        const std::vector<double>& series1, const std::vector<double>& series2,
        double mean1, double mean2) {
        
        double numerator = 0.0;
        double sum_sq_1 = 0.0;
        double sum_sq_2 = 0.0;
        
#ifdef __aarch64__
        // ARM NEON version
        const int simd_size = num_observations / 2;
        
        const float64x2_t vec_mean1 = vdupq_n_f64(mean1);
        const float64x2_t vec_mean2 = vdupq_n_f64(mean2);
        
        float64x2_t vec_numerator = vdupq_n_f64(0.0);
        float64x2_t vec_sum_sq_1 = vdupq_n_f64(0.0);
        float64x2_t vec_sum_sq_2 = vdupq_n_f64(0.0);
        
        for (int k = 0; k < simd_size; k++) {
            const float64x2_t vec_s1 = vld1q_f64(&series1[k * 2]);
            const float64x2_t vec_s2 = vld1q_f64(&series2[k * 2]);
            
            const float64x2_t diff1 = vsubq_f64(vec_s1, vec_mean1);
            const float64x2_t diff2 = vsubq_f64(vec_s2, vec_mean2);
            
            vec_numerator = vfmaq_f64(vec_numerator, diff1, diff2);
            vec_sum_sq_1 = vfmaq_f64(vec_sum_sq_1, diff1, diff1);
            vec_sum_sq_2 = vfmaq_f64(vec_sum_sq_2, diff2, diff2);
        }
        
        numerator = vgetq_lane_f64(vec_numerator, 0) + vgetq_lane_f64(vec_numerator, 1);
        sum_sq_1 = vgetq_lane_f64(vec_sum_sq_1, 0) + vgetq_lane_f64(vec_sum_sq_1, 1);
        sum_sq_2 = vgetq_lane_f64(vec_sum_sq_2, 0) + vgetq_lane_f64(vec_sum_sq_2, 1);
        
        for (int k = simd_size * 2; k < num_observations; k++) {
            const double diff1 = series1[k] - mean1;
            const double diff2 = series2[k] - mean2;
            
            numerator += diff1 * diff2;
            sum_sq_1 += diff1 * diff1;
            sum_sq_2 += diff2 * diff2;
        }
#else
        // AVX version
        const int simd_size = num_observations / 4;
        
        const __m256d vec_mean1 = _mm256_set1_pd(mean1);
        const __m256d vec_mean2 = _mm256_set1_pd(mean2);
        
        __m256d vec_numerator = _mm256_setzero_pd();
        __m256d vec_sum_sq_1 = _mm256_setzero_pd();
        __m256d vec_sum_sq_2 = _mm256_setzero_pd();
        
        for (int k = 0; k < simd_size; k++) {
            const __m256d vec_s1 = _mm256_loadu_pd(&series1[k * 4]);
            const __m256d vec_s2 = _mm256_loadu_pd(&series2[k * 4]);
            
            const __m256d diff1 = _mm256_sub_pd(vec_s1, vec_mean1);
            const __m256d diff2 = _mm256_sub_pd(vec_s2, vec_mean2);
            
            vec_numerator = _mm256_fmadd_pd(diff1, diff2, vec_numerator);
            vec_sum_sq_1 = _mm256_fmadd_pd(diff1, diff1, vec_sum_sq_1);
            vec_sum_sq_2 = _mm256_fmadd_pd(diff2, diff2, vec_sum_sq_2);
        }
        
        double temp[4];
        _mm256_storeu_pd(temp, vec_numerator);
        numerator = temp[0] + temp[1] + temp[2] + temp[3];
        
        _mm256_storeu_pd(temp, vec_sum_sq_1);
        sum_sq_1 = temp[0] + temp[1] + temp[2] + temp[3];
        
        _mm256_storeu_pd(temp, vec_sum_sq_2);
        sum_sq_2 = temp[0] + temp[1] + temp[2] + temp[3];
        
        for (int k = simd_size * 4; k < num_observations; k++) {
            const double diff1 = series1[k] - mean1;
            const double diff2 = series2[k] - mean2;
            
            numerator += diff1 * diff2;
            sum_sq_1 += diff1 * diff1;
            sum_sq_2 += diff2 * diff2;
        }
#endif
        
        double covariance = numerator / (num_observations - 1);
        double correlation = numerator / (std::sqrt(sum_sq_1 * sum_sq_2) + 1e-12);
        
        return {correlation, covariance};
    }
    
public:
    static std::vector<std::vector<double>> generate_correlated_returns(int num_assets, int num_observations, 
                                                                       double base_volatility = 0.02) {
        std::vector<std::vector<double>> returns(num_assets, std::vector<double>(num_observations));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> market_factor(0.0, base_volatility * 0.5);
        std::normal_distribution<double> asset_factor(0.0, base_volatility * 0.8);
        
        for (int obs = 0; obs < num_observations; obs++) {
            double common_market_shock = market_factor(gen);
            
            for (int asset = 0; asset < num_assets; asset++) {
                double asset_specific_shock = asset_factor(gen);
                double sector_correlation = (asset % 3 == 0) ? 0.3 * market_factor(gen) : 0.0;
                
                returns[asset][obs] = common_market_shock + asset_specific_shock + sector_correlation;
            }
        }
        
        return returns;
    }
    
    void validate_correlation_matrix(const std::vector<std::vector<double>>& correlation) {
        bool is_valid = true;
        
        for (int i = 0; i < num_assets; i++) {
            if (std::abs(correlation[i][i] - 1.0) > 1e-6) {
                std::cout << "Warning: Diagonal element [" << i << "][" << i << "] = " 
                          << correlation[i][i] << " (should be 1.0)\n";
                is_valid = false;
            }
        }
        
        for (int i = 0; i < num_assets; i++) {
            for (int j = i + 1; j < num_assets; j++) {
                if (std::abs(correlation[i][j] - correlation[j][i]) > 1e-6) {
                    std::cout << "Warning: Matrix not symmetric at [" << i << "][" << j << "]\n";
                    is_valid = false;
                }
            }
        }
        
        for (int i = 0; i < num_assets; i++) {
            for (int j = 0; j < num_assets; j++) {
                if (correlation[i][j] < -1.0 || correlation[i][j] > 1.0) {
                    std::cout << "Warning: Correlation [" << i << "][" << j << "] = " 
                              << correlation[i][j] << " (outside [-1, 1])\n";
                    is_valid = false;
                }
            }
        }
        
        std::cout << "Correlation matrix validation: " << (is_valid ? "PASSED" : "FAILED") << "\n";
    }
};

int main() {
    std::cout << "SIMD Financial Correlation Engine Benchmark\n\n";
    
    std::vector<std::pair<int, int>> test_configs = {
        {5, 10000},    // Small portfolio, moderate history
        {10, 20000},   // Medium portfolio, long history
        {25, 50000},   // Large portfolio, extensive history
        {50, 100000}   // Very large portfolio, very long history
    };
    
    for (auto [num_assets, num_observations] : test_configs) {
        std::cout << "Testing " << num_assets << " assets with " << num_observations << " observations:\n";
        
        auto returns = FinancialCorrelationEngine::generate_correlated_returns(num_assets, num_observations);
        
        FinancialCorrelationEngine engine(num_assets, num_observations);
        
        // Regular implementation
        auto start = std::chrono::high_resolution_clock::now();
        auto correlation_regular = engine.calculate_correlation_regular(returns);
        auto end = std::chrono::high_resolution_clock::now();
        auto regular_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // SIMD implementation
        start = std::chrono::high_resolution_clock::now();
        auto correlation_simd = engine.calculate_correlation_simd(returns);
        end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double max_diff = 0.0;
        for (int i = 0; i < num_assets; i++) {
            for (int j = 0; j < num_assets; j++) {
                double diff = std::abs(correlation_regular[i][j] - correlation_simd[i][j]);
                max_diff = std::max(max_diff, diff);
            }
        }
        
        double speedup = (double)regular_time.count() / simd_time.count();
        
        std::cout << "  Regular time: " << regular_time.count() << " μs\n";
        std::cout << "  SIMD time: " << simd_time.count() << " μs\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
        std::cout << "  Max difference: " << std::scientific << max_diff << "\n";
        
        if (num_assets <= 10) {
            std::cout << "  Sample correlation matrix:\n";
            for (int i = 0; i < num_assets; i++) {
                std::cout << "    ";
                for (int j = 0; j < num_assets; j++) {
                    std::cout << std::fixed << std::setprecision(3) << correlation_simd[i][j] << " ";
                }
                std::cout << "\n";
            }
        }
        
        engine.validate_correlation_matrix(correlation_simd);
        std::cout << "\n";
    }
    
    return 0;
}
