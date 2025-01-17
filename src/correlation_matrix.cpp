#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#ifdef __aarch64__
    #include <arm_neon.h>
#else
    #include <immintrin.h>
#endif

class CorrelationCalculator {
private:
    int num_assets;
    int num_observations;
    
public:
    CorrelationCalculator(int assets, int observations) 
        : num_assets(assets), num_observations(observations) {}
    
    std::vector<std::vector<double>> calculate_regular(const std::vector<std::vector<double>>& price_data) {
        std::vector<std::vector<double>> correlation(num_assets, std::vector<double>(num_assets, 0.0));
        
        std::vector<double> means(num_assets, 0.0);
        for (int i = 0; i < num_assets; i++) {
            for (int j = 0; j < num_observations; j++) {
                means[i] += price_data[i][j];
            }
            means[i] /= num_observations;
        }
        
        for (int i = 0; i < num_assets; i++) {
            for (int j = i; j < num_assets; j++) {
                double numerator = 0.0;
                double sum_sq_i = 0.0;
                double sum_sq_j = 0.0;
                
                for (int k = 0; k < num_observations; k++) {
                    double diff_i = price_data[i][k] - means[i];
                    double diff_j = price_data[j][k] - means[j];
                    
                    numerator += diff_i * diff_j;
                    sum_sq_i += diff_i * diff_i;
                    sum_sq_j += diff_j * diff_j;
                }
                
                double correlation_coeff = numerator / (sqrt(sum_sq_i * sum_sq_j) + 1e-12);
                correlation[i][j] = correlation_coeff;
                correlation[j][i] = correlation_coeff;  
            }
        }
        
        return correlation;
    }
    
    std::vector<std::vector<double>> calculate_simd(const std::vector<std::vector<double>>& price_data) {
        std::vector<std::vector<double>> correlation(num_assets, std::vector<double>(num_assets, 0.0));
        
        std::vector<double> means(num_assets, 0.0);
        for (int i = 0; i < num_assets; i++) {
            for (int j = 0; j < num_observations; j++) {
                means[i] += price_data[i][j];
            }
            means[i] /= num_observations;
        }
        
        for (int i = 0; i < num_assets; i++) {
            for (int j = i; j < num_assets; j++) {
                double correlation_coeff = calculate_correlation_simd(price_data[i], price_data[j], means[i], means[j]);
                correlation[i][j] = correlation_coeff;
                correlation[j][i] = correlation_coeff;
            }
        }
        
        return correlation;
    }
    
private:
    double calculate_correlation_simd(const std::vector<double>& series1, const std::vector<double>& series2, 
                                     double mean1, double mean2) {
        double numerator = 0.0;
        double sum_sq_1 = 0.0;
        double sum_sq_2 = 0.0;
        
#ifdef __aarch64__
        int simd_size = num_observations / 2;
        
        float64x2_t vec_mean1 = vdupq_n_f64(mean1);
        float64x2_t vec_mean2 = vdupq_n_f64(mean2);
        float64x2_t vec_numerator = vdupq_n_f64(0.0);
        float64x2_t vec_sum_sq_1 = vdupq_n_f64(0.0);
        float64x2_t vec_sum_sq_2 = vdupq_n_f64(0.0);
        
        for (int k = 0; k < simd_size; k++) {
            float64x2_t vec_s1 = vld1q_f64(&series1[k * 2]);
            float64x2_t vec_s2 = vld1q_f64(&series2[k * 2]);
            
            float64x2_t diff1 = vsubq_f64(vec_s1, vec_mean1);
            float64x2_t diff2 = vsubq_f64(vec_s2, vec_mean2);
            
            vec_numerator = vfmaq_f64(vec_numerator, diff1, diff2);
            
            vec_sum_sq_1 = vfmaq_f64(vec_sum_sq_1, diff1, diff1);
            vec_sum_sq_2 = vfmaq_f64(vec_sum_sq_2, diff2, diff2);
        }
        
        numerator = vgetq_lane_f64(vec_numerator, 0) + vgetq_lane_f64(vec_numerator, 1);
        sum_sq_1 = vgetq_lane_f64(vec_sum_sq_1, 0) + vgetq_lane_f64(vec_sum_sq_1, 1);
        sum_sq_2 = vgetq_lane_f64(vec_sum_sq_2, 0) + vgetq_lane_f64(vec_sum_sq_2, 1);
        
        for (int k = simd_size * 2; k < num_observations; k++) {
            double diff1 = series1[k] - mean1;
            double diff2 = series2[k] - mean2;
            
            numerator += diff1 * diff2;
            sum_sq_1 += diff1 * diff1;
            sum_sq_2 += diff2 * diff2;
        }
        
#else
        int simd_size = num_observations / 4;
        
        __m256d vec_mean1 = _mm256_set1_pd(mean1);
        __m256d vec_mean2 = _mm256_set1_pd(mean2);
        __m256d vec_numerator = _mm256_setzero_pd();
        __m256d vec_sum_sq_1 = _mm256_setzero_pd();
        __m256d vec_sum_sq_2 = _mm256_setzero_pd();
        
        for (int k = 0; k < simd_size; k++) {
            __m256d vec_s1 = _mm256_loadu_pd(&series1[k * 4]);
            __m256d vec_s2 = _mm256_loadu_pd(&series2[k * 4]);
            
            __m256d diff1 = _mm256_sub_pd(vec_s1, vec_mean1);
            __m256d diff2 = _mm256_sub_pd(vec_s2, vec_mean2);
            
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
            double diff1 = series1[k] - mean1;
            double diff2 = series2[k] - mean2;
            
            numerator += diff1 * diff2;
            sum_sq_1 += diff1 * diff1;
            sum_sq_2 += diff2 * diff2;
        }
#endif
        
        return numerator / (sqrt(sum_sq_1 * sum_sq_2) + 1e-12);
    }
};

int main() {
    const int NUM_ASSETS = 10;
    const int NUM_OBSERVATIONS = 50000;  
    
    std::cout << "Building correlation matrix for " << NUM_ASSETS << " assets with " 
              << NUM_OBSERVATIONS << " return observations each...\n\n";
    
    std::vector<std::vector<double>> price_data(NUM_ASSETS, std::vector<double>(NUM_OBSERVATIONS));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 0.01);  
    
    for (int i = 0; i < NUM_ASSETS; i++) {
        for (int j = 0; j < NUM_OBSERVATIONS; j++) {
            double base_return = dis(gen);
            double market_factor = (j > 0) ? 0.3 * dis(gen) : 0.0;
            double asset_factor = 0.7 * dis(gen);
            
            price_data[i][j] = base_return + market_factor + asset_factor;
        }
    }
    
    CorrelationCalculator calc(NUM_ASSETS, NUM_OBSERVATIONS);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto correlation_regular = calc.calculate_regular(price_data);
    auto end = std::chrono::high_resolution_clock::now();
    auto regular_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    auto correlation_simd = calc.calculate_simd(price_data);
    end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    bool results_match = true;
    double max_diff = 0.0;
    for (int i = 0; i < NUM_ASSETS; i++) {
        for (int j = 0; j < NUM_ASSETS; j++) {
            double diff = std::abs(correlation_regular[i][j] - correlation_simd[i][j]);
            max_diff = std::max(max_diff, diff);
            if (diff > 1e-10) {
                results_match = false;
            }
        }
    }
    
    std::cout << "Regular calculation time: " << regular_time.count() << " microseconds\n";
    std::cout << "SIMD calculation time: " << simd_time.count() << " microseconds\n";
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) 
              << (double)regular_time.count() / simd_time.count() << "x\n";
    std::cout << "Results match: " << (results_match ? "Yes" : "No") << "\n";
    std::cout << "Max difference: " << std::scientific << max_diff << "\n\n";
    
    std::cout << "Sample correlation matrix (first 5x5):\n";
    for (int i = 0; i < std::min(5, NUM_ASSETS); i++) {
        for (int j = 0; j < std::min(5, NUM_ASSETS); j++) {
            std::cout << std::fixed << std::setprecision(3) << correlation_simd[i][j] << "\t";
        }
        std::cout << "\n";
    }
    
    return 0;
}