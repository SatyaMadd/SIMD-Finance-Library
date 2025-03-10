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
#elif defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#endif

template<typename T>
class AlignedVector {
private:
    T* data_;
    size_t size_;
    size_t capacity_;
    
public:
    AlignedVector(size_t size) : size_(size), capacity_(size) {
        data_ = static_cast<T*>(std::aligned_alloc(32, capacity_ * sizeof(T)));
        if (!data_) throw std::bad_alloc();
    }
    
    ~AlignedVector() {
        if (data_) {
            std::free(data_);
        }
    }
    
    AlignedVector(const AlignedVector& other) : size_(other.size_), capacity_(other.capacity_) {
        data_ = static_cast<T*>(std::aligned_alloc(32, capacity_ * sizeof(T)));
        if (!data_) throw std::bad_alloc();
        std::copy(other.data_, other.data_ + size_, data_);
    }
    
    AlignedVector& operator=(const AlignedVector& other) {
        if (this != &other) {
            if (data_) {
                std::free(data_);
            }
            size_ = other.size_;
            capacity_ = other.capacity_;
            data_ = static_cast<T*>(std::aligned_alloc(32, capacity_ * sizeof(T)));
            if (!data_) throw std::bad_alloc();
            std::copy(other.data_, other.data_ + size_, data_);
        }
        return *this;
    }
    
    AlignedVector(AlignedVector&& other) noexcept : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    
    AlignedVector& operator=(AlignedVector&& other) noexcept {
        if (this != &other) {
            if (data_) {
                std::free(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
};

class SIMDRandomGenerator {
private:
    uint64_t state_[4];  
    static constexpr uint64_t MATRIX_A = 0x9908b0dfUL;
    static constexpr uint64_t UPPER_MASK = 0x80000000UL;
    static constexpr uint64_t LOWER_MASK = 0x7fffffffUL;
    
public:
    SIMDRandomGenerator(uint64_t seed = 5489UL) {
        seed_mt(seed);
    }
    
    void seed_mt(uint64_t seed) {
        state_[0] = seed;
        for (int i = 1; i < 4; i++) {
            state_[i] = (1812433253UL * (state_[i-1] ^ (state_[i-1] >> 30)) + i);
        }
    }
    
    void generate_normal_simd(AlignedVector<double>& output, size_t count) {
        if (count % 4 != 0) {
            throw std::invalid_argument("Count must be divisible by 4 for SIMD");
        }
        
#ifdef __aarch64__
        const float64x2_t two_pi = vdupq_n_f64(2.0 * M_PI);
        const float64x2_t minus_two = vdupq_n_f64(-2.0);
        
        for (size_t i = 0; i < count; i += 4) {
            // Generate 4 uniform random numbers
            float64x2_t u1 = generate_uniform_pair();
            float64x2_t u2 = generate_uniform_pair();
            
            // Box-Muller
            float64x2_t ln_u1 = vlog_f64(u1);  
            float64x2_t sqrt_part = vsqrt_f64(vmulq_f64(minus_two, ln_u1));
            
            float64x2_t angle = vmulq_f64(two_pi, u2);
            float64x2_t cos_angle = vcos_f64(angle);  
            float64x2_t sin_angle = vsin_f64(angle); 
            
            float64x2_t z1 = vmulq_f64(sqrt_part, cos_angle);
            float64x2_t z2 = vmulq_f64(sqrt_part, sin_angle);
            
            vst1q_f64(&output[i], z1);
            vst1q_f64(&output[i + 2], z2);
        }
#else
        // AVX version
        const __m256d two_pi = _mm256_set1_pd(2.0 * M_PI);
        const __m256d minus_two = _mm256_set1_pd(-2.0);
        
        for (size_t i = 0; i < count; i += 4) {
            // Generate 4 uniform random numbers
            __m256d u1 = generate_uniform_avx();
            __m256d u2 = generate_uniform_avx();
            
            // Box-Muller transformation
            __m256d ln_u1 = _mm256_log_pd(u1);  
            __m256d sqrt_part = _mm256_sqrt_pd(_mm256_mul_pd(minus_two, ln_u1));
            
            __m256d angle = _mm256_mul_pd(two_pi, u2);
            __m256d cos_angle = _mm256_cos_pd(angle);  
            
            __m256d normals = _mm256_mul_pd(sqrt_part, cos_angle);
            _mm256_store_pd(&output[i], normals);
        }
#endif
    }
    
    void generate_uniform_batch(AlignedVector<double>& output, size_t count) {
        uint64_t x = state_[0];
        uint64_t y = state_[1];
        uint64_t z = state_[2];
        uint64_t w = state_[3];
        
        for (size_t i = 0; i < count; ++i) {
            uint64_t t = x ^ (x << 11);
            x = y; y = z; z = w;
            w = w ^ (w >> 19) ^ (t ^ (t >> 8));
            
            output[i] = (w >> 11) * (1.0 / 9007199254740992.0);
        }
        
        state_[0] = x; state_[1] = y; state_[2] = z; state_[3] = w;
    }
    
private:
#ifdef __aarch64__
    float64x2_t generate_uniform_pair() {
        uint64_t vals[2];
        for (int i = 0; i < 2; i++) {
            uint64_t t = state_[0] ^ (state_[0] << 11);
            state_[0] = state_[1]; state_[1] = state_[2]; state_[2] = state_[3];
            state_[3] = state_[3] ^ (state_[3] >> 19) ^ (t ^ (t >> 8));
            vals[i] = state_[3];
        }
        
        double dvals[2];
        dvals[0] = (vals[0] >> 11) * (1.0 / 9007199254740992.0);
        dvals[1] = (vals[1] >> 11) * (1.0 / 9007199254740992.0);
        
        return vld1q_f64(dvals);
    }
    
    float64x2_t vlog_f64(float64x2_t x) {
        double vals[2];
        vst1q_f64(vals, x);
        vals[0] = std::log(vals[0]);
        vals[1] = std::log(vals[1]);
        return vld1q_f64(vals);
    }
    
    float64x2_t vcos_f64(float64x2_t x) {
        double vals[2];
        vst1q_f64(vals, x);
        vals[0] = std::cos(vals[0]);
        vals[1] = std::cos(vals[1]);
        return vld1q_f64(vals);
    }
    
    float64x2_t vsin_f64(float64x2_t x) {
        double vals[2];
        vst1q_f64(vals, x);
        vals[0] = std::sin(vals[0]);
        vals[1] = std::sin(vals[1]);
        return vld1q_f64(vals);
    }
    
    float64x2_t vsqrt_f64(float64x2_t x) {
        return vsqrtq_f64(x);
    }
#else
    __m256d generate_uniform_avx() {
        uint64_t vals[4];
        for (int i = 0; i < 4; i++) {
            uint64_t t = state_[0] ^ (state_[0] << 11);
            state_[0] = state_[1]; state_[1] = state_[2]; state_[2] = state_[3];
            state_[3] = state_[3] ^ (state_[3] >> 19) ^ (t ^ (t >> 8));
            vals[i] = state_[3];
        }
        
        double dvals[4];
        for (int i = 0; i < 4; i++) {
            dvals[i] = (vals[i] >> 11) * (1.0 / 9007199254740992.0);
        }
        
        return _mm256_load_pd(dvals);
    }
    
    // Fast transcendental approximations for AVX
    __m256d _mm256_log_pd(__m256d x) {
        double vals[4];
        _mm256_store_pd(vals, x);
        for (int i = 0; i < 4; i++) {
            vals[i] = std::log(vals[i]);
        }
        return _mm256_load_pd(vals);
    }
    
    __m256d _mm256_cos_pd(__m256d x) {
        double vals[4];
        _mm256_store_pd(vals, x);
        for (int i = 0; i < 4; i++) {
            vals[i] = std::cos(vals[i]);
        }
        return _mm256_load_pd(vals);
    }
#endif
};

class FinancialRNG {
private:
    SIMDRandomGenerator rng_;
    AlignedVector<double> buffer_;
    size_t buffer_pos_;
    
public:
    FinancialRNG(size_t buffer_size = 65536) 
        : buffer_(buffer_size), buffer_pos_(buffer_size) {}
    
    void generate_correlated_normals(const std::vector<std::vector<double>>& cholesky_matrix,
                                   std::vector<AlignedVector<double>>& output) {
        int n = cholesky_matrix.size();
        int samples = output[0].size();
        
        std::vector<AlignedVector<double>> independent(n, AlignedVector<double>(samples));
        for (int i = 0; i < n; i++) {
            rng_.generate_normal_simd(independent[i], samples);
        }
        
        for (int i = 0; i < n; i++) {
            for (size_t j = 0; j < samples; j++) {
                output[i][j] = 0.0;
                for (int k = 0; k <= i; k++) {
                    output[i][j] += cholesky_matrix[i][k] * independent[k][j];
                }
            }
        }
    }
    
    void generate_antithetic_normals(AlignedVector<double>& output, size_t count) {
        if (count % 2 != 0) {
            throw std::invalid_argument("Count must be even for antithetic sampling");
        }
        
        AlignedVector<double> temp(count / 2);
        rng_.generate_normal_simd(temp, count / 2);
        
        for (size_t i = 0; i < count / 2; i++) {
            output[i] = temp[i];
            output[i + count / 2] = -temp[i];
        }
    }
};

// Naive/Regular random number generator for comparison
class NaiveRandomGenerator {
private:
    std::mt19937_64 rng_;
    std::normal_distribution<double> normal_dist_;
    std::uniform_real_distribution<double> uniform_dist_;
    
public:
    NaiveRandomGenerator(uint64_t seed = 5489UL) 
        : rng_(seed), normal_dist_(0.0, 1.0), uniform_dist_(0.0, 1.0) {}
    
    void generate_normal_naive(AlignedVector<double>& output, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            output[i] = normal_dist_(rng_);
        }
    }
    
    void generate_uniform_naive(AlignedVector<double>& output, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            output[i] = uniform_dist_(rng_);
        }
    }
    
    // Box-Muller transformation (naive implementation)
    void generate_normal_box_muller(AlignedVector<double>& output, size_t count) {
        if (count % 2 != 0) {
            throw std::invalid_argument("Count must be even for Box-Muller");
        }
        
        for (size_t i = 0; i < count; i += 2) {
            double u1 = uniform_dist_(rng_);
            double u2 = uniform_dist_(rng_);
            
            double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            double z1 = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);
            
            output[i] = z0;
            output[i + 1] = z1;
        }
    }
};

void benchmark_random_generation() {
    const size_t test_size = 1000000;
    
    std::cout << "\n=== SIMD vs Naive Random Number Generation Benchmark ===\n\n";
    
    std::vector<size_t> sizes = {10000, 100000, 1000000, 5000000};
    
    for (size_t size : sizes) {
        std::cout << "Testing with " << size << " random numbers:\n";
        
        AlignedVector<double> output_simd(size);
        AlignedVector<double> output_naive(size);
        AlignedVector<double> output_box_muller(size);
        
        size_t adjusted_size = (size / 4) * 4;
        
        SIMDRandomGenerator simd_rng(12345);
        NaiveRandomGenerator naive_rng(12345);
        NaiveRandomGenerator box_muller_rng(12345);
        
        auto start = std::chrono::high_resolution_clock::now();
        simd_rng.generate_normal_simd(output_simd, adjusted_size);
        auto end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        naive_rng.generate_normal_naive(output_naive, size);
        end = std::chrono::high_resolution_clock::now();
        auto naive_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        box_muller_rng.generate_normal_box_muller(output_box_muller, (size / 2) * 2);
        end = std::chrono::high_resolution_clock::now();
        auto box_muller_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout << "  SIMD time: " << simd_time << " μs\n";
        std::cout << "  Naive time: " << naive_time << " μs\n";
        std::cout << "  Box-Muller time: " << box_muller_time << " μs\n";
        std::cout << "  SIMD vs Naive speedup: " << std::fixed << std::setprecision(2) 
                  << (double)naive_time / simd_time << "x\n";
        std::cout << "  SIMD vs Box-Muller speedup: " << std::fixed << std::setprecision(2) 
                  << (double)box_muller_time / simd_time << "x\n";
        
        double simd_mean = 0.0, naive_mean = 0.0, box_muller_mean = 0.0;
        double simd_var = 0.0, naive_var = 0.0, box_muller_var = 0.0;
        
        for (size_t i = 0; i < adjusted_size; ++i) {
            simd_mean += output_simd[i];
        }
        simd_mean /= adjusted_size;
        
        for (size_t i = 0; i < size; ++i) {
            naive_mean += output_naive[i];
        }
        naive_mean /= size;
        
        for (size_t i = 0; i < (size / 2) * 2; ++i) {
            box_muller_mean += output_box_muller[i];
        }
        box_muller_mean /= (size / 2) * 2;
        
        for (size_t i = 0; i < adjusted_size; ++i) {
            simd_var += (output_simd[i] - simd_mean) * (output_simd[i] - simd_mean);
        }
        simd_var /= (adjusted_size - 1);
        
        for (size_t i = 0; i < size; ++i) {
            naive_var += (output_naive[i] - naive_mean) * (output_naive[i] - naive_mean);
        }
        naive_var /= (size - 1);
        
        for (size_t i = 0; i < (size / 2) * 2; ++i) {
            box_muller_var += (output_box_muller[i] - box_muller_mean) * (output_box_muller[i] - box_muller_mean);
        }
        box_muller_var /= ((size / 2) * 2 - 1);
        
        std::cout << "  Statistical Properties:\n";
        std::cout << "    SIMD - Mean: " << std::fixed << std::setprecision(6) << simd_mean 
                  << ", Variance: " << simd_var << "\n";
        std::cout << "    Naive - Mean: " << std::fixed << std::setprecision(6) << naive_mean 
                  << ", Variance: " << naive_var << "\n";
        std::cout << "    Box-Muller - Mean: " << std::fixed << std::setprecision(6) << box_muller_mean 
                  << ", Variance: " << box_muller_var << "\n";
        std::cout << "    Expected - Mean: 0.0, Variance: 1.0\n";
        
        double simd_mean_error = std::abs(simd_mean) * 100.0;
        double naive_mean_error = std::abs(naive_mean) * 100.0;
        double simd_var_error = std::abs(simd_var - 1.0) * 100.0;
        double naive_var_error = std::abs(naive_var - 1.0) * 100.0;
        
        std::cout << "  Accuracy:\n";
        std::cout << "    SIMD mean error: " << std::fixed << std::setprecision(4) << simd_mean_error << "%\n";
        std::cout << "    Naive mean error: " << std::fixed << std::setprecision(4) << naive_mean_error << "%\n";
        std::cout << "    SIMD variance error: " << std::fixed << std::setprecision(4) << simd_var_error << "%\n";
        std::cout << "    Naive variance error: " << std::fixed << std::setprecision(4) << naive_var_error << "%\n\n";
    }
    
    std::cout << "=== Uniform Random Number Generation Benchmark ===\n";
    const size_t uniform_size = 1000000;
    
    AlignedVector<double> uniform_simd(uniform_size);
    AlignedVector<double> uniform_naive(uniform_size);
    
    SIMDRandomGenerator simd_rng_uniform(12345);
    NaiveRandomGenerator naive_rng_uniform(12345);
    
    auto start = std::chrono::high_resolution_clock::now();
    simd_rng_uniform.generate_uniform_batch(uniform_simd, uniform_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_uniform_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    naive_rng_uniform.generate_uniform_naive(uniform_naive, uniform_size);
    end = std::chrono::high_resolution_clock::now();
    auto naive_uniform_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Uniform Generation (" << uniform_size << " numbers):\n";
    std::cout << "  SIMD time: " << simd_uniform_time << " μs\n";
    std::cout << "  Naive time: " << naive_uniform_time << " μs\n";
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
              << (double)naive_uniform_time / simd_uniform_time << "x\n\n";
}

int main() {
    try {
        benchmark_random_generation();
        
        std::cout << "=== Correlated Random Number Generation Demo ===\n";
        
        std::vector<std::vector<double>> corr_matrix = {
            {1.0, 0.5, 0.3},
            {0.5, 1.0, 0.4},
            {0.3, 0.4, 1.0}
        };
        
        std::vector<std::vector<double>> cholesky = {
            {1.0, 0.0, 0.0},
            {0.5, 0.866, 0.0},
            {0.3, 0.231, 0.919}
        };
        
        const size_t num_samples = 10000;
        std::vector<AlignedVector<double>> correlated_samples(3, AlignedVector<double>(num_samples));
        
        FinancialRNG fin_rng;
        
        auto start = std::chrono::high_resolution_clock::now();
        fin_rng.generate_correlated_normals(cholesky, correlated_samples);
        auto end = std::chrono::high_resolution_clock::now();
        auto corr_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout << "Correlated random generation time: " << corr_time << " μs\n";
        
        std::vector<double> means(3, 0.0);
        for (int i = 0; i < 3; i++) {
            for (size_t j = 0; j < num_samples; j++) {
                means[i] += correlated_samples[i][j];
            }
            means[i] /= num_samples;
        }
        
        std::cout << "Sample correlation matrix:\n";
        std::cout << "        Asset1    Asset2    Asset3\n";
        for (int i = 0; i < 3; i++) {
            std::cout << "Asset" << (i+1) << "  ";
            for (int j = 0; j < 3; j++) {
                if (j <= i) {
                    double correlation = 0.0;
                    double var_i = 0.0, var_j = 0.0;
                    
                    for (size_t k = 0; k < num_samples; k++) {
                        double diff_i = correlated_samples[i][k] - means[i];
                        double diff_j = correlated_samples[j][k] - means[j];
                        correlation += diff_i * diff_j;
                        var_i += diff_i * diff_i;
                        var_j += diff_j * diff_j;
                    }
                    
                    correlation /= sqrt(var_i * var_j);
                    std::cout << std::fixed << std::setprecision(3) << std::setw(8) << correlation << "  ";
                } else {
                    std::cout << "        ";
                }
            }
            std::cout << "\n";
        }
        
        std::cout << "\nExpected correlation matrix:\n";
        std::cout << "        Asset1    Asset2    Asset3\n";
        for (int i = 0; i < 3; i++) {
            std::cout << "Asset" << (i+1) << "  ";
            for (int j = 0; j < 3; j++) {
                if (j <= i) {
                    std::cout << std::fixed << std::setprecision(3) << std::setw(8) << corr_matrix[i][j] << "  ";
                } else {
                    std::cout << "        ";
                }
            }
            std::cout << "\n";
        }
        
        std::cout << "\n=== Antithetic Sampling Demo ===\n";
        const size_t antithetic_size = 10000;
        AlignedVector<double> antithetic_samples(antithetic_size);
        
        start = std::chrono::high_resolution_clock::now();
        fin_rng.generate_antithetic_normals(antithetic_samples, antithetic_size);
        end = std::chrono::high_resolution_clock::now();
        auto antithetic_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout << "Antithetic sampling time: " << antithetic_time << " μs\n";
        
        double sum_first_half = 0.0, sum_second_half = 0.0;
        for (size_t i = 0; i < antithetic_size / 2; i++) {
            sum_first_half += antithetic_samples[i];
            sum_second_half += antithetic_samples[i + antithetic_size / 2];
        }
        
        std::cout << "Sum of first half: " << std::fixed << std::setprecision(6) << sum_first_half << "\n";
        std::cout << "Sum of second half: " << std::fixed << std::setprecision(6) << sum_second_half << "\n";
        std::cout << "Sum of both halves: " << std::fixed << std::setprecision(6) << sum_first_half + sum_second_half << "\n";
        std::cout << "Expected sum: 0.0 (due to antithetic property)\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
