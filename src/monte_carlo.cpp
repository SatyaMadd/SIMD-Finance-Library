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

class MonteCarloSimulator {
private:
    int num_paths;
    int num_steps;
    double dt;
    double drift;
    double volatility;
    double initial_price;
    
    std::vector<double> random_normals;
    std::vector<double> path_buffer;
    
    std::mt19937_64 rng;
    mutable std::uniform_real_distribution<double> uniform_dist;
    
public:
    MonteCarloSimulator(int paths, int steps, double time_horizon, 
                       double drift_rate, double vol, double S0)
        : num_paths(paths), num_steps(steps), dt(time_horizon / steps),
          drift(drift_rate), volatility(vol), initial_price(S0),
          rng(std::chrono::steady_clock::now().time_since_epoch().count()),
          uniform_dist(0.0, 1.0) {
        
        random_normals.reserve(num_paths * num_steps);
        path_buffer.reserve(num_paths * num_steps);
    }
    
    void generate_random_normals_regular() {
        random_normals.clear();
        random_normals.reserve(num_paths * num_steps);
        
        for (int i = 0; i < num_paths * num_steps; i += 2) {
            double u1 = uniform_dist(rng);
            double u2 = uniform_dist(rng);
            
            double z1 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            double z2 = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);
            
            random_normals.push_back(z1);
            if (i + 1 < num_paths * num_steps) {
                random_normals.push_back(z2);
            }
        }
    }
    
    void generate_random_normals_simd() {
        random_normals.clear();
        random_normals.reserve(num_paths * num_steps);
        
        const int total_randoms = num_paths * num_steps;
        const int simd_batch = 4; 
        
#ifdef __aarch64__
        for (int i = 0; i < total_randoms; i += simd_batch) {
            double u1_vals[2], u2_vals[2];
            for (int j = 0; j < 2; j++) {
                u1_vals[j] = uniform_dist(rng);
                u2_vals[j] = uniform_dist(rng);
            }
            
            double sqrt_parts[2], angles[2];
            for (int j = 0; j < 2; j++) {
                sqrt_parts[j] = std::sqrt(-2.0 * std::log(u1_vals[j]));
                angles[j] = 2.0 * M_PI * u2_vals[j];
            }
            
            float64x2_t sqrt_vec = vld1q_f64(sqrt_parts);
            
            double cos_vals[2], sin_vals[2];
            for (int j = 0; j < 2; j++) {
                cos_vals[j] = std::cos(angles[j]);
                sin_vals[j] = std::sin(angles[j]);
            }
            
            float64x2_t cos_vec = vld1q_f64(cos_vals);
            float64x2_t sin_vec = vld1q_f64(sin_vals);
            
            float64x2_t z1 = vmulq_f64(sqrt_vec, cos_vec);
            float64x2_t z2 = vmulq_f64(sqrt_vec, sin_vec);
            
            double results[4];
            vst1q_f64(&results[0], z1);
            vst1q_f64(&results[2], z2);
            
            for (int j = 0; j < 4 && i + j < total_randoms; j++) {
                random_normals.push_back(results[j]);
            }
        }
#else
        for (int i = 0; i < total_randoms; i += simd_batch) {
            double u1_vals[2], u2_vals[2];
            for (int j = 0; j < 2; j++) {
                u1_vals[j] = uniform_dist(rng);
                u2_vals[j] = uniform_dist(rng);
            }
            
            double sqrt_parts[2], angles[2];
            for (int j = 0; j < 2; j++) {
                sqrt_parts[j] = std::sqrt(-2.0 * std::log(u1_vals[j]));
                angles[j] = 2.0 * M_PI * u2_vals[j];
            }
            
            double cos_vals[2], sin_vals[2];
            for (int j = 0; j < 2; j++) {
                cos_vals[j] = std::cos(angles[j]);
                sin_vals[j] = std::sin(angles[j]);
            }
            
            __m128d sqrt_vec = _mm_load_pd(sqrt_parts);
            __m128d cos_vec = _mm_load_pd(cos_vals);
            __m128d sin_vec = _mm_load_pd(sin_vals);
            
            __m128d z1 = _mm_mul_pd(sqrt_vec, cos_vec);
            __m128d z2 = _mm_mul_pd(sqrt_vec, sin_vec);
            
            double results[4];
            _mm_store_pd(&results[0], z1);
            _mm_store_pd(&results[2], z2);
            
            for (int j = 0; j < 4 && i + j < total_randoms; j++) {
                random_normals.push_back(results[j]);
            }
        }
#endif
    }
    
    std::vector<std::vector<double>> simulate_paths_regular() {
        generate_random_normals_regular();
        
        std::vector<std::vector<double>> paths(num_paths, std::vector<double>(num_steps + 1));
        
        for (int i = 0; i < num_paths; i++) {
            paths[i][0] = initial_price;
        }
        
        for (int path = 0; path < num_paths; path++) {
            for (int step = 0; step < num_steps; step++) {
                double random_shock = random_normals[path * num_steps + step];
                double drift_term = (drift - 0.5 * volatility * volatility) * dt;
                double diffusion_term = volatility * std::sqrt(dt) * random_shock;
                
                paths[path][step + 1] = paths[path][step] * 
                                       std::exp(drift_term + diffusion_term);
            }
        }
        
        return paths;
    }
    
    std::vector<std::vector<double>> simulate_paths_simd() {
        generate_random_normals_simd();
        
        std::vector<std::vector<double>> paths(num_paths, std::vector<double>(num_steps + 1));
        
        for (int i = 0; i < num_paths; i++) {
            paths[i][0] = initial_price;
        }
        
        const double drift_term = (drift - 0.5 * volatility * volatility) * dt;
        const double vol_sqrt_dt = volatility * std::sqrt(dt);
        
#ifdef __aarch64__
        // ARM NEON version
        const float64x2_t drift_vec = vdupq_n_f64(drift_term);
        const float64x2_t vol_vec = vdupq_n_f64(vol_sqrt_dt);
        
        for (int path = 0; path < num_paths; path += 2) {
            for (int step = 0; step < num_steps; step++) {
                float64x2_t current_prices = {paths[path][step], 
                                            (path + 1 < num_paths) ? paths[path + 1][step] : paths[path][step]};
                
                float64x2_t random_shocks = {random_normals[path * num_steps + step],
                                           (path + 1 < num_paths) ? random_normals[(path + 1) * num_steps + step] : 0.0};
                
                float64x2_t diffusion = vmulq_f64(vol_vec, random_shocks);
                
                float64x2_t exponent = vaddq_f64(drift_vec, diffusion);
                
                double exp_vals[2];
                exp_vals[0] = std::exp(vgetq_lane_f64(exponent, 0));
                exp_vals[1] = std::exp(vgetq_lane_f64(exponent, 1));
                float64x2_t exp_vec = vld1q_f64(exp_vals);
                
                float64x2_t new_prices = vmulq_f64(current_prices, exp_vec);
                
                paths[path][step + 1] = vgetq_lane_f64(new_prices, 0);
                if (path + 1 < num_paths) {
                    paths[path + 1][step + 1] = vgetq_lane_f64(new_prices, 1);
                }
            }
        }
#else
        // AVX version
        const __m256d drift_vec = _mm256_set1_pd(drift_term);
        const __m256d vol_vec = _mm256_set1_pd(vol_sqrt_dt);
        
        for (int path = 0; path < num_paths; path += 4) {
            for (int step = 0; step < num_steps; step++) {
                double current_prices_array[4];
                double random_shocks_array[4];
                
                for (int i = 0; i < 4; i++) {
                    int p = path + i;
                    current_prices_array[i] = (p < num_paths) ? paths[p][step] : paths[path][step];
                    random_shocks_array[i] = (p < num_paths) ? random_normals[p * num_steps + step] : 0.0;
                }
                
                __m256d current_prices = _mm256_load_pd(current_prices_array);
                __m256d random_shocks = _mm256_load_pd(random_shocks_array);
                
                __m256d diffusion = _mm256_mul_pd(vol_vec, random_shocks);
                __m256d exponent = _mm256_add_pd(drift_vec, diffusion);
                
                double exp_vals[4];
                _mm256_store_pd(exp_vals, exponent);
                for (int i = 0; i < 4; i++) {
                    exp_vals[i] = std::exp(exp_vals[i]);
                }
                __m256d exp_vec = _mm256_load_pd(exp_vals);
                
                __m256d new_prices = _mm256_mul_pd(current_prices, exp_vec);
                
                double new_prices_array[4];
                _mm256_store_pd(new_prices_array, new_prices);
                
                for (int i = 0; i < 4; i++) {
                    int p = path + i;
                    if (p < num_paths) {
                        paths[p][step + 1] = new_prices_array[i];
                    }
                }
            }
        }
#endif
        
        return paths;
    }
    
    std::vector<double> calculate_call_payoffs(const std::vector<std::vector<double>>& paths, 
                                             double strike) {
        std::vector<double> payoffs(num_paths);
        
        for (int i = 0; i < num_paths; i++) {
            double final_price = paths[i][num_steps];
            payoffs[i] = std::max(final_price - strike, 0.0);
        }
        
        return payoffs;
    }
    
    void print_statistics(const std::vector<double>& payoffs) {
        double sum = 0.0;
        double sum_sq = 0.0;
        
        for (double payoff : payoffs) {
            sum += payoff;
            sum_sq += payoff * payoff;
        }
        
        double mean = sum / num_paths;
        double variance = (sum_sq / num_paths) - (mean * mean);
        double std_dev = std::sqrt(variance);
        double std_error = std_dev / std::sqrt(num_paths);
        
        std::cout << "  Option price estimate: " << std::fixed << std::setprecision(4) << mean << "\n";
        std::cout << "  Standard error: " << std::fixed << std::setprecision(6) << std_error << "\n";
        std::cout << "  95% confidence interval: [" 
                  << mean - 1.96 * std_error << ", " 
                  << mean + 1.96 * std_error << "]\n";
    }
};

int main() {
    std::cout << "SIMD Monte Carlo Options Pricing Benchmark\n\n";
    
    const double S0 = 100.0;        // Initial stock price
    const double K = 105.0;         // Strike price
    const double r = 0.05;          // Risk-free rate
    const double sigma = 0.2;       // Volatility
    const double T = 1.0;           // Time to maturity
    
    std::vector<int> path_counts = {10000, 50000, 100000, 500000};
    const int num_steps = 252;     
    
    for (int num_paths : path_counts) {
        std::cout << "Testing with " << num_paths << " paths, " << num_steps << " steps:\n";
        
        MonteCarloSimulator simulator(num_paths, num_steps, T, r, sigma, S0);
        
        // Regular implementation
        auto start = std::chrono::high_resolution_clock::now();
        auto paths_regular = simulator.simulate_paths_regular();
        auto payoffs_regular = simulator.calculate_call_payoffs(paths_regular, K);
        auto end = std::chrono::high_resolution_clock::now();
        auto regular_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // SIMD implementation
        start = std::chrono::high_resolution_clock::now();
        auto paths_simd = simulator.simulate_paths_simd();
        auto payoffs_simd = simulator.calculate_call_payoffs(paths_simd, K);
        end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        double speedup = (double)regular_time.count() / simd_time.count();
        
        std::cout << "  Regular time: " << regular_time.count() << " ms\n";
        std::cout << "  SIMD time: " << simd_time.count() << " ms\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
        
        std::cout << "  Regular results:\n";
        simulator.print_statistics(payoffs_regular);
        std::cout << "  SIMD results:\n";
        simulator.print_statistics(payoffs_simd);
        
        double regular_mean = 0.0, simd_mean = 0.0;
        for (int i = 0; i < num_paths; i++) {
            regular_mean += payoffs_regular[i];
            simd_mean += payoffs_simd[i];
        }
        regular_mean /= num_paths;
        simd_mean /= num_paths;
        
        double error = std::abs(regular_mean - simd_mean) / regular_mean * 100.0;
        std::cout << "  Relative error: " << std::fixed << std::setprecision(3) << error << "%\n\n";
    }
    
    return 0;
}
