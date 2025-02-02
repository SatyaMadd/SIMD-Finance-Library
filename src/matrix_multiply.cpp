#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cstring>

#ifdef __aarch64__
    #include <arm_neon.h>
#else
    #include <immintrin.h>
#endif

class Matrix {
private:
    std::vector<double> data;
    int rows, cols;
    
public:
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(rows * cols, 0.0);
    }
    
    double& operator()(int r, int c) {
        return data[r * cols + c];
    }
    
    const double& operator()(int r, int c) const {
        return data[r * cols + c];
    }
    
    double* raw_data() { return data.data(); }
    const double* raw_data() const { return data.data(); }
    
    int get_rows() const { return rows; }
    int get_cols() const { return cols; }
    
    void fill_random() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        
        for (auto& val : data) {
            val = dis(gen);
        }
    }
    
    void fill_test_pattern() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                (*this)(i, j) = i * cols + j + 1;
            }
        }
    }
    
    bool is_equal(const Matrix& other, double tolerance = 1e-10) const {
        if (rows != other.rows || cols != other.cols) return false;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (std::abs((*this)(i, j) - other(i, j)) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
};

class MatrixMultiplier {
public:
    static Matrix multiply_regular(const Matrix& A, const Matrix& B) {
        int m = A.get_rows();
        int n = A.get_cols();
        int p = B.get_cols();
        
        if (n != B.get_rows()) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        
        Matrix C(m, p);
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
        
        return C;
    }
    
    static Matrix multiply_simd(const Matrix& A, const Matrix& B) {
        int m = A.get_rows();
        int n = A.get_cols();
        int p = B.get_cols();
        
        if (n != B.get_rows()) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        
        Matrix C(m, p);
        
        const int BLOCK_SIZE = 64;
        
        for (int ii = 0; ii < m; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < p; jj += BLOCK_SIZE) {
                for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
                    
                    int i_max = std::min(ii + BLOCK_SIZE, m);
                    int j_max = std::min(jj + BLOCK_SIZE, p);
                    int k_max = std::min(kk + BLOCK_SIZE, n);
                    
                    for (int i = ii; i < i_max; i++) {
                        for (int j = jj; j < j_max; j += 4) {  
                            
                            int j_limit = std::min(j + 4, j_max);
                            
#ifdef __aarch64__
                            float64x2_t sum1 = vdupq_n_f64(0.0);
                            float64x2_t sum2 = vdupq_n_f64(0.0);
                            
                            for (int k = kk; k < k_max; k++) {
                                float64x2_t a_val = vdupq_n_f64(A(i, k));
                                
                                if (j + 1 < j_limit) {
                                    float64x2_t b_vals = {B(k, j), B(k, j + 1)};
                                    sum1 = vfmaq_f64(sum1, a_val, b_vals);
                                }
                                
                                if (j + 3 < j_limit) {
                                    float64x2_t b_vals2 = {B(k, j + 2), B(k, j + 3)};
                                    sum2 = vfmaq_f64(sum2, a_val, b_vals2);
                                }
                            }
                            
                            if (j < j_limit) C(i, j) += vgetq_lane_f64(sum1, 0);
                            if (j + 1 < j_limit) C(i, j + 1) += vgetq_lane_f64(sum1, 1);
                            if (j + 2 < j_limit) C(i, j + 2) += vgetq_lane_f64(sum2, 0);
                            if (j + 3 < j_limit) C(i, j + 3) += vgetq_lane_f64(sum2, 1);
#else
                            __m256d sum = _mm256_setzero_pd();
                            
                            for (int k = kk; k < k_max; k++) {
                                __m256d a_val = _mm256_set1_pd(A(i, k));
                                
                                if (j + 3 < j_limit) {
                                    __m256d b_vals = _mm256_set_pd(B(k, j + 3), B(k, j + 2), 
                                                                  B(k, j + 1), B(k, j));
                                    sum = _mm256_fmadd_pd(a_val, b_vals, sum);
                                }
                            }                            
                            double temp[4];
                            _mm256_storeu_pd(temp, sum);
                            
                            if (j < j_limit) C(i, j) += temp[0];
                            if (j + 1 < j_limit) C(i, j + 1) += temp[1];
                            if (j + 2 < j_limit) C(i, j + 2) += temp[2];
                            if (j + 3 < j_limit) C(i, j + 3) += temp[3];
#endif
                        }
                    }
                }
            }
        }
        
        return C;
    }
    
    static Matrix multiply_simd_square(const Matrix& A, const Matrix& B) {
        int n = A.get_rows();
        
        if (n != A.get_cols() || n != B.get_rows() || n != B.get_cols()) {
            throw std::invalid_argument("Matrices must be square and same size");
        }
        
        Matrix C(n, n);
        
        const int BLOCK_SIZE = 32;
        
        for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
                for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
                    
                    int i_max = std::min(ii + BLOCK_SIZE, n);
                    int j_max = std::min(jj + BLOCK_SIZE, n);
                    int k_max = std::min(kk + BLOCK_SIZE, n);
                    
                    for (int i = ii; i < i_max; i++) {
                        
#ifdef __aarch64__
                        for (int j = jj; j < j_max; j += 2) {
                            if (j + 1 >= j_max) break;
                            
                            float64x2_t sum = vdupq_n_f64(0.0);
                            
                            for (int k = kk; k < k_max; k++) {
                                float64x2_t a_val = vdupq_n_f64(A(i, k));
                                float64x2_t b_vals = {B(k, j), B(k, j + 1)};
                                sum = vfmaq_f64(sum, a_val, b_vals);
                            }
                            
                            C(i, j) += vgetq_lane_f64(sum, 0);
                            C(i, j + 1) += vgetq_lane_f64(sum, 1);
                        }
#else
                        for (int j = jj; j < j_max; j += 4) {
                            if (j + 3 >= j_max) break;
                            
                            __m256d sum = _mm256_setzero_pd();
                            
                            for (int k = kk; k < k_max; k++) {
                                __m256d a_val = _mm256_set1_pd(A(i, k));
                                __m256d b_vals = _mm256_set_pd(B(k, j + 3), B(k, j + 2), 
                                                              B(k, j + 1), B(k, j));
                                sum = _mm256_fmadd_pd(a_val, b_vals, sum);
                            }
                            
                            double temp[4];
                            _mm256_storeu_pd(temp, sum);
                            
                            C(i, j) += temp[0];
                            C(i, j + 1) += temp[1];
                            C(i, j + 2) += temp[2];
                            C(i, j + 3) += temp[3];
                        }
#endif
                    }
                }
            }
        }
        
        return C;
    }
};

int main() {
    std::cout << "SIMD Matrix Multiplication Benchmark\n\n";

    std::vector<int> sizes = {64, 128, 256, 512};
    
    for (int size : sizes) {
        std::cout << "Testing " << size << "x" << size << " matrices:\n";
        
        Matrix A(size, size);
        Matrix B(size, size);
        
        A.fill_random();
        B.fill_random();
        
        auto start = std::chrono::high_resolution_clock::now();
        Matrix C_regular = MatrixMultiplier::multiply_regular(A, B);
        auto end = std::chrono::high_resolution_clock::now();
        auto regular_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        Matrix C_simd = MatrixMultiplier::multiply_simd_square(A, B);
        end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        bool correct = C_regular.is_equal(C_simd, 1e-10);
        
        std::cout << "  Regular time: " << regular_time.count() << " ms\n";
        std::cout << "  SIMD time: " << simd_time.count() << " ms\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
                  << (double)regular_time.count() / simd_time.count() << "x\n";
        std::cout << "  Results match: " << (correct ? "Yes" : "No") << "\n\n";
    }
    
    std::cout << "Small example verification:\n";
    Matrix A_small(3, 3);
    Matrix B_small(3, 3);
    
    A_small.fill_test_pattern();
    B_small.fill_test_pattern();
    
    std::cout << "Matrix A:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << A_small(i, j) << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nMatrix B:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << B_small(i, j) << " ";
        }
        std::cout << "\n";
    }
    
    Matrix C_result = MatrixMultiplier::multiply_simd_square(A_small, B_small);
    std::cout << "\nResult C = A * B:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << C_result(i, j) << " ";
        }
        std::cout << "\n";
    }
    
    return 0;
}