#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <random>

#ifdef __aarch64__
    #include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#endif

class SIMDBlackScholes {
private:
    static double cdf_normal_approx(double x) {
        const double a1 =  0.254829592;
        const double a2 = -0.284496736;
        const double a3 =  1.421413741;
        const double a4 = -1.453152027;
        const double a5 =  1.061405429;
        const double p  =  0.3275911;
        
        double sign = (x >= 0) ? 1.0 : -1.0;
        x = std::abs(x) / std::sqrt(2.0);
        
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);
        
        return 0.5 * (1.0 + sign * y);
    }
    
#ifdef __aarch64__
    static float64x2_t fast_log_neon(float64x2_t x) {
        float64x2_t one = vdupq_n_f64(1.0);
        float64x2_t ln2 = vdupq_n_f64(0.693147180559945309417);
        
        float64x2_t u = vsubq_f64(x, one);
        float64x2_t u2 = vmulq_f64(u, u);
        float64x2_t six = vdupq_n_f64(6.0);
        float64x2_t four = vdupq_n_f64(4.0);
        
        float64x2_t numerator = vmulq_f64(u, vaddq_f64(six, u));
        float64x2_t denominator = vfmaq_f64(vfmaq_f64(six, four, u), one, u2);
        
        return vdivq_f64(numerator, denominator);
    }
    
    static float64x2_t fast_sqrt_neon(float64x2_t x) {
        return vsqrtq_f64(x);
    }
    
    static float64x2_t fast_exp_neon(float64x2_t x) {
        
        float64x2_t one = vdupq_n_f64(1.0);
        float64x2_t half = vdupq_n_f64(0.5);
        
        float64x2_t x_clamped = vmaxq_f64(vminq_f64(x, vdupq_n_f64(10.0)), vdupq_n_f64(-10.0));
        
        float64x2_t x_half = vmulq_f64(x_clamped, half);
        float64x2_t numerator = vaddq_f64(one, x_half);
        float64x2_t denominator = vsubq_f64(one, x_half);
        
        return vdivq_f64(numerator, denominator);
    }
    
    static float64x2_t fast_cdf_normal_neon(float64x2_t x) {
        const float64x2_t a1 = vdupq_n_f64(0.254829592);
        const float64x2_t a2 = vdupq_n_f64(-0.284496736);
        const float64x2_t a3 = vdupq_n_f64(1.421413741);
        const float64x2_t a4 = vdupq_n_f64(-1.453152027);
        const float64x2_t a5 = vdupq_n_f64(1.061405429);
        const float64x2_t p = vdupq_n_f64(0.3275911);
        const float64x2_t half = vdupq_n_f64(0.5);
        const float64x2_t one = vdupq_n_f64(1.0);
        const float64x2_t sqrt2 = vdupq_n_f64(1.41421356237);
        
        float64x2_t sign = vbslq_f64(vcgeq_f64(x, vdupq_n_f64(0.0)), one, vdupq_n_f64(-1.0));
        
        x = vabsq_f64(vdivq_f64(x, sqrt2));
        
        float64x2_t t = vdivq_f64(one, vfmaq_f64(one, p, x));
        float64x2_t poly = vfmaq_f64(a1, a2, t);
        poly = vfmaq_f64(poly, a3, t);
        poly = vfmaq_f64(poly, a4, t);
        poly = vfmaq_f64(poly, a5, t);
        
        float64x2_t neg_x_sq = vnegq_f64(vmulq_f64(x, x));
        float64x2_t exp_term = fast_exp_neon(neg_x_sq);
        
        float64x2_t y = vsubq_f64(one, vmulq_f64(vmulq_f64(poly, t), exp_term));
        
        return vmulq_f64(half, vfmaq_f64(one, sign, y));
    }
#else
    // Fast vectorized functions for AVX
    static __m256d fast_log_avx(__m256d x) {
        __m256d one = _mm256_set1_pd(1.0);
        __m256d six = _mm256_set1_pd(6.0);
        __m256d four = _mm256_set1_pd(4.0);
        
        __m256d u = _mm256_sub_pd(x, one);
        __m256d u2 = _mm256_mul_pd(u, u);
        
        __m256d numerator = _mm256_mul_pd(u, _mm256_add_pd(six, u));
        __m256d denominator = _mm256_fmadd_pd(u2, one, _mm256_fmadd_pd(u, four, six));
        
        return _mm256_div_pd(numerator, denominator);
    }
    
    static __m256d fast_sqrt_avx(__m256d x) {
        return _mm256_sqrt_pd(x);
    }
    
    static __m256d fast_exp_avx(__m256d x) {
        __m256d one = _mm256_set1_pd(1.0);
        __m256d half = _mm256_set1_pd(0.5);
        __m256d max_val = _mm256_set1_pd(10.0);
        __m256d min_val = _mm256_set1_pd(-10.0);
        
        __m256d x_clamped = _mm256_max_pd(_mm256_min_pd(x, max_val), min_val);
        
        __m256d x_half = _mm256_mul_pd(x_clamped, half);
        __m256d numerator = _mm256_add_pd(one, x_half);
        __m256d denominator = _mm256_sub_pd(one, x_half);
        
        return _mm256_div_pd(numerator, denominator);
    }
    
    static __m256d fast_cdf_normal_avx(__m256d x) {
        const __m256d a1 = _mm256_set1_pd(0.254829592);
        const __m256d a2 = _mm256_set1_pd(-0.284496736);
        const __m256d a3 = _mm256_set1_pd(1.421413741);
        const __m256d a4 = _mm256_set1_pd(-1.453152027);
        const __m256d a5 = _mm256_set1_pd(1.061405429);
        const __m256d p = _mm256_set1_pd(0.3275911);
        const __m256d half = _mm256_set1_pd(0.5);
        const __m256d one = _mm256_set1_pd(1.0);
        const __m256d sqrt2 = _mm256_set1_pd(1.41421356237);
        const __m256d neg_one = _mm256_set1_pd(-1.0);
        
        __m256d sign = _mm256_blendv_pd(neg_one, one, _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GE_OQ));
        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
        abs_x = _mm256_div_pd(abs_x, sqrt2);
        
        __m256d t = _mm256_div_pd(one, _mm256_fmadd_pd(p, abs_x, one));
        
        __m256d poly = _mm256_fmadd_pd(a5, t, a4);
        poly = _mm256_fmadd_pd(poly, t, a3);
        poly = _mm256_fmadd_pd(poly, t, a2);
        poly = _mm256_fmadd_pd(poly, t, a1);
        
        __m256d neg_x_sq = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_mul_pd(abs_x, abs_x));
        __m256d exp_term = fast_exp_avx(neg_x_sq);
        
        __m256d y = _mm256_sub_pd(one, _mm256_mul_pd(_mm256_mul_pd(poly, t), exp_term));
        
        return _mm256_mul_pd(half, _mm256_fmadd_pd(sign, y, one));
    }
#endif
    
public:
    static double black_scholes_call_regular(double S, double K, double T, double r, double sigma) {
        if (T <= 0) return std::max(S - K, 0.0);
        
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        
        double call_price = S * cdf_normal_approx(d1) - K * std::exp(-r * T) * cdf_normal_approx(d2);
        
        return call_price;
    }
    
    static double black_scholes_put_regular(double S, double K, double T, double r, double sigma) {
        if (T <= 0) return std::max(K - S, 0.0);
        
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        
        double put_price = K * std::exp(-r * T) * cdf_normal_approx(-d2) - S * cdf_normal_approx(-d1);
        
        return put_price;
    }
    
    static void black_scholes_call_naive_batch(const std::vector<double>& S, const std::vector<double>& K,
                                               const std::vector<double>& T, const std::vector<double>& r,
                                               const std::vector<double>& sigma, std::vector<double>& prices) {
        size_t n = S.size();
        prices.resize(n);
        
        if (n != K.size() || n != T.size() || n != r.size() || n != sigma.size()) {
            throw std::invalid_argument("All input vectors must have the same size");
        }
        
#ifdef __aarch64__
        const size_t batch_size = 2;
#else
        const size_t batch_size = 4;
#endif
        
        size_t batch_end = (n / batch_size) * batch_size;
        
        for (size_t i = 0; i < batch_end; i += batch_size) {
            for (size_t j = 0; j < batch_size; j++) {
                size_t idx = i + j;
                
                if (T[idx] <= 0) {
                    prices[idx] = std::max(S[idx] - K[idx], 0.0);
                    continue;
                }
                
                double d1 = (std::log(S[idx] / K[idx]) + (r[idx] + 0.5 * sigma[idx] * sigma[idx]) * T[idx]) / (sigma[idx] * std::sqrt(T[idx]));
                double d2 = d1 - sigma[idx] * std::sqrt(T[idx]);
                
                prices[idx] = S[idx] * cdf_normal_approx(d1) - K[idx] * std::exp(-r[idx] * T[idx]) * cdf_normal_approx(d2);
            }
        }
        
        for (size_t i = batch_end; i < n; i++) {
            prices[i] = black_scholes_call_regular(S[i], K[i], T[i], r[i], sigma[i]);
        }
    }
    static void black_scholes_call_simd(const std::vector<double>& S, const std::vector<double>& K,
                                       const std::vector<double>& T, const std::vector<double>& r,
                                       const std::vector<double>& sigma, std::vector<double>& prices) {
        size_t n = S.size();
        prices.resize(n);
        
        if (n != K.size() || n != T.size() || n != r.size() || n != sigma.size()) {
            throw std::invalid_argument("All input vectors must have the same size");
        }
        
#ifdef __aarch64__
        size_t simd_end = (n / 2) * 2;
        
        for (size_t i = 0; i < simd_end; i += 2) {
            float64x2_t S_vec = vld1q_f64(&S[i]);
            float64x2_t K_vec = vld1q_f64(&K[i]);
            float64x2_t T_vec = vld1q_f64(&T[i]);
            float64x2_t r_vec = vld1q_f64(&r[i]);
            float64x2_t sigma_vec = vld1q_f64(&sigma[i]);
            
            float64x2_t S_over_K = vdivq_f64(S_vec, K_vec);
            float64x2_t log_S_over_K = fast_log_neon(S_over_K);
            
            float64x2_t half = vdupq_n_f64(0.5);
            float64x2_t sigma_sq = vmulq_f64(sigma_vec, sigma_vec);
            float64x2_t drift = vfmaq_f64(r_vec, half, sigma_sq);
            float64x2_t drift_T = vmulq_f64(drift, T_vec);
            float64x2_t numerator = vaddq_f64(log_S_over_K, drift_T);
            
            float64x2_t sqrt_T = fast_sqrt_neon(T_vec);
            float64x2_t sigma_sqrt_T = vmulq_f64(sigma_vec, sqrt_T);
            float64x2_t d1 = vdivq_f64(numerator, sigma_sqrt_T);
            float64x2_t d2 = vsubq_f64(d1, sigma_sqrt_T);
            
            float64x2_t N_d1 = fast_cdf_normal_neon(d1);
            float64x2_t N_d2 = fast_cdf_normal_neon(d2);
            
            float64x2_t neg_rT = vnegq_f64(vmulq_f64(r_vec, T_vec));
            float64x2_t discount = fast_exp_neon(neg_rT);
            
            float64x2_t first_term = vmulq_f64(S_vec, N_d1);
            float64x2_t second_term = vmulq_f64(vmulq_f64(K_vec, discount), N_d2);
            float64x2_t call_price = vsubq_f64(first_term, second_term);
            
            vst1q_f64(&prices[i], call_price);
        }
        
        for (size_t i = simd_end; i < n; i++) {
            prices[i] = black_scholes_call_regular(S[i], K[i], T[i], r[i], sigma[i]);
        }
        
#else
        size_t simd_end = (n / 4) * 4;
        
        for (size_t i = 0; i < simd_end; i += 4) {
            __m256d S_vec = _mm256_loadu_pd(&S[i]);
            __m256d K_vec = _mm256_loadu_pd(&K[i]);
            __m256d T_vec = _mm256_loadu_pd(&T[i]);
            __m256d r_vec = _mm256_loadu_pd(&r[i]);
            __m256d sigma_vec = _mm256_loadu_pd(&sigma[i]);
            
            __m256d S_over_K = _mm256_div_pd(S_vec, K_vec);
            __m256d log_S_over_K = fast_log_avx(S_over_K);
            
            __m256d half = _mm256_set1_pd(0.5);
            __m256d sigma_sq = _mm256_mul_pd(sigma_vec, sigma_vec);
            __m256d drift = _mm256_fmadd_pd(half, sigma_sq, r_vec);
            __m256d drift_T = _mm256_mul_pd(drift, T_vec);
            __m256d numerator = _mm256_add_pd(log_S_over_K, drift_T);
            
            __m256d sqrt_T = fast_sqrt_avx(T_vec);
            __m256d sigma_sqrt_T = _mm256_mul_pd(sigma_vec, sqrt_T);
            __m256d d1 = _mm256_div_pd(numerator, sigma_sqrt_T);
            __m256d d2 = _mm256_sub_pd(d1, sigma_sqrt_T);
            
            __m256d N_d1 = fast_cdf_normal_avx(d1);
            __m256d N_d2 = fast_cdf_normal_avx(d2);
            
            __m256d neg_rT = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_mul_pd(r_vec, T_vec));
            __m256d discount = fast_exp_avx(neg_rT);
            
            __m256d first_term = _mm256_mul_pd(S_vec, N_d1);
            __m256d second_term = _mm256_mul_pd(_mm256_mul_pd(K_vec, discount), N_d2);
            __m256d call_price = _mm256_sub_pd(first_term, second_term);
            
            _mm256_storeu_pd(&prices[i], call_price);
        }
        
        for (size_t i = simd_end; i < n; i++) {
            prices[i] = black_scholes_call_regular(S[i], K[i], T[i], r[i], sigma[i]);
        }
#endif
    }
    
    struct Greeks {
        double delta;
        double gamma;
        double theta;
        double vega;
        double rho;
    };
    
    static Greeks calculate_greeks_regular(double S, double K, double T, double r, double sigma) {
        Greeks greeks;
        
        if (T <= 0) {
            greeks.delta = (S > K) ? 1.0 : 0.0;
            greeks.gamma = 0.0;
            greeks.theta = 0.0;
            greeks.vega = 0.0;
            greeks.rho = 0.0;
            return greeks;
        }
        
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        
        double N_d1 = cdf_normal_approx(d1);
        double N_d2 = cdf_normal_approx(d2);
        double n_d1 = std::exp(-0.5 * d1 * d1) / std::sqrt(2 * M_PI);
        
        greeks.delta = N_d1;
        greeks.gamma = n_d1 / (S * sigma * std::sqrt(T));
        greeks.theta = -(S * n_d1 * sigma) / (2 * std::sqrt(T)) - r * K * std::exp(-r * T) * N_d2;
        greeks.vega = S * n_d1 * std::sqrt(T);
        greeks.rho = K * T * std::exp(-r * T) * N_d2;
        
        return greeks;
    }
    
    static void calculate_greeks_simd(const std::vector<double>& S, const std::vector<double>& K,
                                     const std::vector<double>& T, const std::vector<double>& r,
                                     const std::vector<double>& sigma, std::vector<Greeks>& greeks) {
        size_t n = S.size();
        greeks.resize(n);
        for (size_t i = 0; i < n; i++) {
            greeks[i] = calculate_greeks_regular(S[i], K[i], T[i], r[i], sigma[i]);
        }
    }
    
    static void analyze_accuracy(const std::vector<double>& S, const std::vector<double>& K,
                                const std::vector<double>& T, const std::vector<double>& r,
                                const std::vector<double>& sigma, int sample_size = 1000) {
        std::cout << "\n=== SIMD Accuracy Analysis ===" << std::endl;
        
        std::vector<double> simd_prices, naive_prices;
        
        size_t n = std::min(sample_size, static_cast<int>(S.size()));
        std::vector<double> test_S(S.begin(), S.begin() + n);
        std::vector<double> test_K(K.begin(), K.begin() + n);
        std::vector<double> test_T(T.begin(), T.begin() + n);
        std::vector<double> test_r(r.begin(), r.begin() + n);
        std::vector<double> test_sigma(sigma.begin(), sigma.begin() + n);
        
        black_scholes_call_simd(test_S, test_K, test_T, test_r, test_sigma, simd_prices);
        black_scholes_call_naive_batch(test_S, test_K, test_T, test_r, test_sigma, naive_prices);
        
        double max_abs_error = 0.0;
        double max_rel_error = 0.0;
        double avg_abs_error = 0.0;
        double avg_rel_error = 0.0;
        
        for (size_t i = 0; i < n; i++) {
            double abs_error = std::abs(simd_prices[i] - naive_prices[i]);
            double rel_error = abs_error / std::max(naive_prices[i], 1e-10);
            
            max_abs_error = std::max(max_abs_error, abs_error);
            max_rel_error = std::max(max_rel_error, rel_error);
            avg_abs_error += abs_error;
            avg_rel_error += rel_error;
        }
        
        avg_abs_error /= n;
        avg_rel_error /= n;
        
        std::cout << "Sample size: " << n << " options" << std::endl;
        std::cout << "Maximum absolute error: $" << std::fixed << std::setprecision(4) << max_abs_error << std::endl;
        std::cout << "Average absolute error: $" << std::fixed << std::setprecision(4) << avg_abs_error << std::endl;
        std::cout << "Maximum relative error: " << std::fixed << std::setprecision(4) << (max_rel_error * 100) << "%" << std::endl;
        std::cout << "Average relative error: " << std::fixed << std::setprecision(4) << (avg_rel_error * 100) << "%" << std::endl;
        
        int low_error = 0, medium_error = 0, high_error = 0;
        for (size_t i = 0; i < n; i++) {
            double rel_error = std::abs(simd_prices[i] - naive_prices[i]) / std::max(naive_prices[i], 1e-10);
            if (rel_error < 0.01) low_error++;
            else if (rel_error < 0.05) medium_error++;
            else high_error++;
        }
        
        std::cout << "\nError Distribution:" << std::endl;
        std::cout << "Low error (<1%): " << low_error << " options (" << (low_error * 100.0 / n) << "%)" << std::endl;
        std::cout << "Medium error (1-5%): " << medium_error << " options (" << (medium_error * 100.0 / n) << "%)" << std::endl;
        std::cout << "High error (>5%): " << high_error << " options (" << (high_error * 100.0 / n) << "%)" << std::endl;
    }
};

class OptionPortfolio {
private:
    std::vector<double> spot_prices_;
    std::vector<double> strike_prices_;
    std::vector<double> times_to_expiry_;
    std::vector<double> risk_free_rates_;
    std::vector<double> volatilities_;
    std::vector<double> positions_; 
    
public:
    void add_option(double S, double K, double T, double r, double sigma, double position) {
        spot_prices_.push_back(S);
        strike_prices_.push_back(K);
        times_to_expiry_.push_back(T);
        risk_free_rates_.push_back(r);
        volatilities_.push_back(sigma);
        positions_.push_back(position);
    }
    
    double calculate_portfolio_value() {
        if (spot_prices_.empty()) return 0.0;
        
        std::vector<double> option_prices;
        SIMDBlackScholes::black_scholes_call_simd(spot_prices_, strike_prices_, times_to_expiry_,
                                                  risk_free_rates_, volatilities_, option_prices);
        
        double total_value = 0.0;
        for (size_t i = 0; i < option_prices.size(); i++) {
            total_value += positions_[i] * option_prices[i];
        }
        
        return total_value;
    }
    
    SIMDBlackScholes::Greeks calculate_portfolio_greeks() {
        if (spot_prices_.empty()) {
            return {0.0, 0.0, 0.0, 0.0, 0.0};
        }
        
        std::vector<SIMDBlackScholes::Greeks> option_greeks;
        SIMDBlackScholes::calculate_greeks_simd(spot_prices_, strike_prices_, times_to_expiry_,
                                               risk_free_rates_, volatilities_, option_greeks);
        
        SIMDBlackScholes::Greeks portfolio_greeks = {0.0, 0.0, 0.0, 0.0, 0.0};
        
        for (size_t i = 0; i < option_greeks.size(); i++) {
            portfolio_greeks.delta += positions_[i] * option_greeks[i].delta;
            portfolio_greeks.gamma += positions_[i] * option_greeks[i].gamma;
            portfolio_greeks.theta += positions_[i] * option_greeks[i].theta;
            portfolio_greeks.vega += positions_[i] * option_greeks[i].vega;
            portfolio_greeks.rho += positions_[i] * option_greeks[i].rho;
        }
        
        return portfolio_greeks;
    }
    
    double calculate_delta_var(double spot_volatility, double confidence_level, double time_horizon) {
        auto portfolio_greeks = calculate_portfolio_greeks();
        
        double z_score = (confidence_level == 0.95) ? 1.645 : 
                        (confidence_level == 0.99) ? 2.326 : 1.96;
        
        double avg_spot = 0.0;
        for (double spot : spot_prices_) {
            avg_spot += spot;
        }
        avg_spot /= spot_prices_.size();
        
        double portfolio_volatility = std::abs(portfolio_greeks.delta) * avg_spot * spot_volatility;
        
        return z_score * portfolio_volatility * std::sqrt(time_horizon);
    }
    
    size_t size() const { return spot_prices_.size(); }
    
    void clear() {
        spot_prices_.clear();
        strike_prices_.clear();
        times_to_expiry_.clear();
        risk_free_rates_.clear();
        volatilities_.clear();
        positions_.clear();
    }
};

class NaiveBlackScholes {
public:
    static double black_scholes_call(double S, double K, double T, double r, double sigma) {
        if (T <= 0) return std::max(S - K, 0.0);
        
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        
        double call_price = S * cdf_normal_approx(d1) - K * std::exp(-r * T) * cdf_normal_approx(d2);
        
        return call_price;
    }
    
    static double black_scholes_put(double S, double K, double T, double r, double sigma) {
        if (T <= 0) return std::max(K - S, 0.0);
        
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        
        double put_price = K * std::exp(-r * T) * cdf_normal_approx(-d2) - S * cdf_normal_approx(-d1);
        
        return put_price;
    }
    
    static void batch_pricing(const std::vector<double>& S, const std::vector<double>& K,
                             const std::vector<double>& T, const std::vector<double>& r,
                             const std::vector<double>& sigma, std::vector<double>& call_prices,
                             std::vector<double>& put_prices) {
        size_t n = S.size();
        call_prices.resize(n);
        put_prices.resize(n);
        
        for (size_t i = 0; i < n; i++) {
            call_prices[i] = black_scholes_call(S[i], K[i], T[i], r[i], sigma[i]);
            put_prices[i] = black_scholes_put(S[i], K[i], T[i], r[i], sigma[i]);
        }
    }

private:
    static double cdf_normal_approx(double x) {
        const double a1 =  0.254829592;
        const double a2 = -0.284496736;
        const double a3 =  1.421413741;
        const double a4 = -1.453152027;
        const double a5 =  1.061405429;
        const double p  =  0.3275911;
        
        double sign = (x >= 0) ? 1.0 : -1.0;
        x = std::abs(x) / std::sqrt(2.0);
        
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);
        
        return 0.5 * (1.0 + sign * y);
    }
};

class ExtendedSIMDBlackScholes : public SIMDBlackScholes {
public:
    static void black_scholes_put_simd(const std::vector<double>& S, const std::vector<double>& K,
                                      const std::vector<double>& T, const std::vector<double>& r,
                                      const std::vector<double>& sigma, std::vector<double>& prices) {
        size_t n = S.size();
        prices.resize(n);
        
        if (n != K.size() || n != T.size() || n != r.size() || n != sigma.size()) {
            throw std::invalid_argument("All input vectors must have the same size");
        }
        
        for (size_t i = 0; i < n; i++) {
            prices[i] = black_scholes_put_regular(S[i], K[i], T[i], r[i], sigma[i]);
        }
    }
    
    static void batch_pricing_simd(const std::vector<double>& S, const std::vector<double>& K,
                                  const std::vector<double>& T, const std::vector<double>& r,
                                  const std::vector<double>& sigma, std::vector<double>& call_prices,
                                  std::vector<double>& put_prices) {
        black_scholes_call_simd(S, K, T, r, sigma, call_prices);
        black_scholes_put_simd(S, K, T, r, sigma, put_prices);
    }
};

void benchmark_black_scholes_pricing() {
    std::cout << "\n=== Black-Scholes Option Pricing Benchmark ===\n\n";
    
    std::vector<size_t> batch_sizes = {1000, 10000, 100000, 500000};
    
    for (size_t batch_size : batch_sizes) {
        std::cout << "Testing with " << batch_size << " options:\n";
        
        std::vector<double> S(batch_size);     // Stock prices
        std::vector<double> K(batch_size);     // Strike prices
        std::vector<double> T(batch_size);     // Time to expiration
        std::vector<double> r(batch_size);     // Risk-free rate
        std::vector<double> sigma(batch_size); // Volatility
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> s_dist(80.0, 120.0);
        std::uniform_real_distribution<double> k_dist(90.0, 110.0);
        std::uniform_real_distribution<double> t_dist(0.1, 1.0);
        std::uniform_real_distribution<double> r_dist(0.01, 0.1);
        std::uniform_real_distribution<double> vol_dist(0.1, 0.4);
        
        for (size_t i = 0; i < batch_size; ++i) {
            S[i] = s_dist(gen);
            K[i] = k_dist(gen);
            T[i] = t_dist(gen);
            r[i] = r_dist(gen);
            sigma[i] = vol_dist(gen);
        }
        
        std::vector<double> call_prices_simd(batch_size);
        std::vector<double> call_prices_naive(batch_size);
        
        auto start = std::chrono::high_resolution_clock::now();
        SIMDBlackScholes::black_scholes_call_simd(S, K, T, r, sigma, call_prices_simd);
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        SIMDBlackScholes::black_scholes_call_naive_batch(S, K, T, r, sigma, call_prices_naive);
        end = std::chrono::high_resolution_clock::now();
        auto naive_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        double speedup = simd_time > 0 ? (double)naive_time / simd_time : 0.0;
        
        std::cout << "  SIMD time: " << simd_time << " μs\n";
        std::cout << "  Naive time: " << naive_time << " μs\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
        
        double max_call_error = 0.0;
        double avg_call_error = 0.0;
        
        for (size_t i = 0; i < batch_size; ++i) {
            double call_error = std::abs(call_prices_simd[i] - call_prices_naive[i]);
            
            max_call_error = std::max(max_call_error, call_error);
            avg_call_error += call_error;
        }
        
        avg_call_error /= batch_size;
        
        std::cout << "  Accuracy:\n";
        std::cout << "    Call max error: " << std::scientific << std::setprecision(2) << max_call_error << "\n";
        std::cout << "    Call avg error: " << std::scientific << std::setprecision(2) << avg_call_error << "\n";
        
        if (batch_size <= 1000) {
            std::cout << "  Sample Results (first 3 options):\n";
            for (size_t i = 0; i < std::min(size_t(3), batch_size); ++i) {
                std::cout << "    Option " << (i+1) << " - S:" << std::fixed << std::setprecision(2) << S[i] 
                         << " K:" << K[i] << " T:" << T[i] << "\n";
                std::cout << "      SIMD Call: " << std::setprecision(4) << call_prices_simd[i] << "\n";
                std::cout << "      Naive Call: " << call_prices_naive[i] << "\n";
            }
        }
        std::cout << "\n";
        
        if (batch_size == 1000) {
            SIMDBlackScholes::analyze_accuracy(S, K, T, r, sigma, 1000);
        }
    }
}

void benchmark_greeks_calculation() {
    std::cout << "=== Greeks Calculation Benchmark ===\n\n";
    
    std::vector<size_t> batch_sizes = {1000, 10000, 100000};
    
    for (size_t batch_size : batch_sizes) {
        std::cout << "Testing Greeks calculation with " << batch_size << " options:\n";
        
        std::vector<double> S(batch_size, 100.0);
        std::vector<double> K(batch_size, 105.0);
        std::vector<double> T(batch_size, 0.25);
        std::vector<double> r(batch_size, 0.05);
        std::vector<double> sigma(batch_size, 0.2);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> s_dist(90.0, 110.0);
        std::uniform_real_distribution<double> vol_dist(0.15, 0.35);
        
        for (size_t i = 0; i < batch_size; ++i) {
            S[i] = s_dist(gen);
            sigma[i] = vol_dist(gen);
        }
        
        std::vector<SIMDBlackScholes::Greeks> greeks_simd(batch_size);
        std::vector<SIMDBlackScholes::Greeks> greeks_naive(batch_size);
        
        auto start = std::chrono::high_resolution_clock::now();
        SIMDBlackScholes::calculate_greeks_simd(S, K, T, r, sigma, greeks_simd);
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < batch_size; ++i) {
            greeks_naive[i] = SIMDBlackScholes::calculate_greeks_regular(S[i], K[i], T[i], r[i], sigma[i]);
        }
        end = std::chrono::high_resolution_clock::now();
        auto naive_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        double speedup = simd_time > 0 ? (double)naive_time / simd_time : 0.0;
        
        std::cout << "  SIMD time: " << simd_time << " μs\n";
        std::cout << "  Naive time: " << naive_time << " μs\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n\n";
    }
}

void benchmark_portfolio_scenarios() {
    std::cout << "=== Portfolio Scenario Analysis ===\n\n";
    
    std::vector<size_t> portfolio_sizes = {10, 100, 1000, 5000};
    
    for (size_t size : portfolio_sizes) {
        std::cout << "Portfolio with " << size << " positions:\n";
        
        OptionPortfolio portfolio;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> s_dist(80.0, 120.0);
        std::uniform_real_distribution<double> k_dist(70.0, 130.0);
        std::uniform_real_distribution<double> t_dist(0.1, 2.0);
        std::uniform_real_distribution<double> vol_dist(0.1, 0.5);
        std::uniform_int_distribution<int> pos_dist(-1000, 1000);
        
        for (size_t i = 0; i < size; ++i) {
            portfolio.add_option(s_dist(gen), k_dist(gen), t_dist(gen), 0.05, vol_dist(gen), pos_dist(gen));
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        double portfolio_value = portfolio.calculate_portfolio_value();
        auto end = std::chrono::high_resolution_clock::now();
        auto value_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        auto greeks = portfolio.calculate_portfolio_greeks();
        end = std::chrono::high_resolution_clock::now();
        auto greeks_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        double var_95 = portfolio.calculate_delta_var(0.2, 0.95, 1.0/252.0);
        end = std::chrono::high_resolution_clock::now();
        auto var_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout << "  Portfolio valuation: " << value_time << " μs\n";
        std::cout << "  Greeks calculation: " << greeks_time << " μs\n";
        std::cout << "  VaR calculation: " << var_time << " μs\n";
        std::cout << "  Total time: " << (value_time + greeks_time + var_time) << " μs\n";
        std::cout << "  Portfolio value: $" << std::fixed << std::setprecision(2) << portfolio_value << "\n";
        std::cout << "  Portfolio delta: " << std::setprecision(4) << greeks.delta << "\n";
        std::cout << "  1-day 95% VaR: $" << std::setprecision(2) << var_95 << "\n\n";
        
        portfolio.clear();
    }
}

int main() {
    std::cout << "=== SIMD vs Naive Black-Scholes Performance Analysis ===\n";
    
    benchmark_black_scholes_pricing();
    benchmark_greeks_calculation();
    benchmark_portfolio_scenarios();
    
    std::cout << "=== Option Portfolio Example ===\n\n";
    
    OptionPortfolio portfolio;
    
    portfolio.add_option(100.0, 105.0, 0.25, 0.05, 0.2, 100);    // Long 100 calls
    portfolio.add_option(100.0, 95.0, 0.25, 0.05, 0.2, -50);     // Short 50 puts
    portfolio.add_option(110.0, 115.0, 0.5, 0.05, 0.25, 200);    // Long 200 calls 
    portfolio.add_option(90.0, 85.0, 0.75, 0.05, 0.3, -75);      // Short 75 puts 
    portfolio.add_option(105.0, 100.0, 0.1, 0.05, 0.15, 300);    // Long 300 calls 
    
    std::cout << "Portfolio contains " << portfolio.size() << " option positions\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    double portfolio_value = portfolio.calculate_portfolio_value();
    auto end = std::chrono::high_resolution_clock::now();
    auto value_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    auto greeks = portfolio.calculate_portfolio_greeks();
    end = std::chrono::high_resolution_clock::now();
    auto greeks_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Portfolio value: $" << std::fixed << std::setprecision(2) << portfolio_value << "\n";
    std::cout << "Valuation time: " << value_time << " μs\n";
    std::cout << "Greeks calculation time: " << greeks_time << " μs\n\n";
    
    std::cout << "Portfolio Greeks:\n";
    std::cout << "  Delta: " << std::fixed << std::setprecision(4) << greeks.delta << " (price sensitivity)\n";
    std::cout << "  Gamma: " << std::fixed << std::setprecision(4) << greeks.gamma << " (delta sensitivity)\n";
    std::cout << "  Theta: " << std::fixed << std::setprecision(4) << greeks.theta << " (time decay)\n";
    std::cout << "  Vega:  " << std::fixed << std::setprecision(4) << greeks.vega << " (volatility sensitivity)\n";
    std::cout << "  Rho:   " << std::fixed << std::setprecision(4) << greeks.rho << " (interest rate sensitivity)\n\n";
    
    std::vector<double> time_horizons = {1.0/252.0, 1.0/52.0, 1.0/12.0};  
    std::vector<std::string> horizon_names = {"1-day", "1-week", "1-month"};
    
    std::cout << "Value at Risk (95% confidence):\n";
    for (size_t i = 0; i < time_horizons.size(); ++i) {
        double var_95 = portfolio.calculate_delta_var(0.2, 0.95, time_horizons[i]);
        std::cout << "  " << horizon_names[i] << " VaR: $" << std::fixed << std::setprecision(2) << var_95 << "\n";
    }
    return 0;
}
