#include <iostream>
#include <chrono>
#ifdef __aarch64__
    #include <arm_neon.h>  
#else
    #include <immintrin.h>  
#endif
#include <vector>
#include <random>

void multiply_regular(const float* a, const float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}
void multiply_simd(const float* a, const float* b, float* result, int size) {
#ifdef __aarch64__
    int simd_size = size / 4;
    for (int i = 0; i < simd_size; i++) {
        float32x4_t vec_a = vld1q_f32(&a[i * 4]);
        float32x4_t vec_b = vld1q_f32(&b[i * 4]);        
        float32x4_t vec_result = vmulq_f32(vec_a, vec_b);        
        vst1q_f32(&result[i * 4], vec_result);
    }
    
    for (int i = simd_size * 4; i < size; i++) {
        result[i] = a[i] * b[i];
    }
#else
    int simd_size = size / 8;
    
    for (int i = 0; i < simd_size; i++) {
        __m256 vec_a = _mm256_load_ps(&a[i * 8]);
        __m256 vec_b = _mm256_load_ps(&b[i * 8]);
        
        __m256 vec_result = _mm256_mul_ps(vec_a, vec_b);
        
        _mm256_store_ps(&result[i * 8], vec_result);
    }
    
    for (int i = simd_size * 8; i < size; i++) {
        result[i] = a[i] * b[i];
    }
#endif
}

int main() {
    const int SIZE = 1000000;  
    
    std::vector<float> a(SIZE);
    std::vector<float> b(SIZE);
    std::vector<float> result_regular(SIZE);
    std::vector<float> result_simd(SIZE);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    
    for (int i = 0; i < SIZE; i++) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    multiply_regular(a.data(), b.data(), result_regular.data(), SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    auto regular_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    multiply_simd(a.data(), b.data(), result_simd.data(), SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    bool correct = true;
    for (int i = 0; i < SIZE; i++) {
        if (std::abs(result_regular[i] - result_simd[i]) > 0.001f) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Array size: " << SIZE << " elements\n";
    std::cout << "Regular time: " << regular_time.count() << " microseconds\n";
    std::cout << "SIMD time: " << simd_time.count() << " microseconds\n";
    std::cout << "Speedup: " << (double)regular_time.count() / simd_time.count() << "x\n";
    std::cout << "Results match: " << (correct ? "Yes" : "No") << "\n";
    
    return 0;
}