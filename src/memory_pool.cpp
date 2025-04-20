#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <random>
#include <iomanip>
#include <unordered_map>
#include <thread>
#include <mutex>

#ifdef _WIN32
    #define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
    #define aligned_free(ptr) _aligned_free(ptr)
#else
    #define aligned_free(ptr) free(ptr)
#endif

class MemoryPool {
private:
    struct Block {
        char* data;
        size_t size;
        size_t allocated_size;  
        bool is_free;
        Block* next;
        
        Block(size_t s) : size(s), allocated_size(0), is_free(true), next(nullptr) {
            data = static_cast<char*>(aligned_alloc(32, size));
            if (!data) throw std::bad_alloc();
        }
        
        ~Block() {
            if (data) {
                aligned_free(data);
            }
        }
    };
    
    std::vector<std::unique_ptr<Block>> large_blocks_;
    std::vector<std::unique_ptr<Block>> small_blocks_;
    Block* free_list_head_;
    
    static constexpr size_t SMALL_BLOCK_SIZE = 4096;    
    static constexpr size_t LARGE_BLOCK_SIZE = 1048576; 
    static constexpr size_t ALIGNMENT = 32;             
    
    size_t total_allocated_;
    size_t total_used_;
    size_t peak_used_;       
    size_t allocation_count_;
    
    size_t align_size(size_t size) {
        return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }
    
    Block* find_best_fit(size_t size) {
        Block* best = nullptr;
        size_t best_size = SIZE_MAX;
        
        for (auto& block : small_blocks_) {
            if (block->is_free && block->size >= size && block->size < best_size) {
                best = block.get();
                best_size = block->size;
            }
        }
        
        for (auto& block : large_blocks_) {
            if (block->is_free && block->size >= size && block->size < best_size) {
                best = block.get();
                best_size = block->size;
            }
        }
        
        return best;
    }
    
    Block* allocate_new_block(size_t size) {
        size_t block_size = std::max(size, SMALL_BLOCK_SIZE);
        
        if (block_size <= SMALL_BLOCK_SIZE) {
            small_blocks_.push_back(std::make_unique<Block>(SMALL_BLOCK_SIZE));
            total_allocated_ += SMALL_BLOCK_SIZE;
            return small_blocks_.back().get();
        } else {
            large_blocks_.push_back(std::make_unique<Block>(block_size));
            total_allocated_ += block_size;
            return large_blocks_.back().get();
        }
    }
    
public:
    MemoryPool() : free_list_head_(nullptr), total_allocated_(0), total_used_(0), peak_used_(0), allocation_count_(0) {}
    
    ~MemoryPool() {
        std::cout << "Memory Pool Statistics:\n";
        std::cout << "  Total allocated: " << total_allocated_ << " bytes\n";
        std::cout << "  Peak usage: " << peak_used_ << " bytes\n";
        std::cout << "  Allocation count: " << allocation_count_ << "\n";
        std::cout << "  Efficiency: " << std::fixed << std::setprecision(2) 
                  << (total_allocated_ > 0 ? (double)peak_used_ / total_allocated_ * 100.0 : 0.0) << "%\n";
    }
    
    void* allocate(size_t size) {
        if (size == 0) return nullptr;
        
        size_t aligned_size = align_size(size);
        Block* block = find_best_fit(aligned_size);
        
        if (!block) {
            block = allocate_new_block(aligned_size);
        }
        
        block->is_free = false;
        block->allocated_size = aligned_size;
        total_used_ += aligned_size;
        peak_used_ = std::max(peak_used_, total_used_);
        allocation_count_++;
        
        return block->data;
    }
    
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        for (auto& block : small_blocks_) {
            if (block->data == ptr) {
                if (!block->is_free) {
                    block->is_free = true;
                    total_used_ -= block->allocated_size;
                    block->allocated_size = 0;
                }
                return;
            }
        }
        
        for (auto& block : large_blocks_) {
            if (block->data == ptr) {
                if (!block->is_free) {
                    block->is_free = true;
                    total_used_ -= block->allocated_size;
                    block->allocated_size = 0;
                }
                return;
            }
        }
    }
    
    std::vector<void*> batch_allocate(size_t size, size_t count) {
        std::vector<void*> ptrs;
        ptrs.reserve(count);
        
        for (size_t i = 0; i < count; i++) {
            ptrs.push_back(allocate(size));
        }
        
        return ptrs;
    }
    
    size_t get_total_allocated() const { return total_allocated_; }
    size_t get_total_used() const { return total_used_; }
    size_t get_allocation_count() const { return allocation_count_; }
    double get_efficiency() const { 
        return total_allocated_ > 0 ? (double)peak_used_ / total_allocated_ : 0.0; 
    }
    
    void reset() {
        for (auto& block : small_blocks_) {
            if (!block->is_free) {
                total_used_ -= block->allocated_size;
                block->allocated_size = 0;
            }
            block->is_free = true;
        }
        for (auto& block : large_blocks_) {
            if (!block->is_free) {
                total_used_ -= block->allocated_size;
                block->allocated_size = 0;
            }
            block->is_free = true;
        }
    }
};

template<typename T>
class PoolAllocator {
private:
    MemoryPool* pool_;
    
public:
    using value_type = T;
    
    PoolAllocator(MemoryPool* pool) : pool_(pool) {}
    
    template<typename U>
    PoolAllocator(const PoolAllocator<U>& other) : pool_(other.pool_) {}
    
    T* allocate(size_t n) {
        return static_cast<T*>(pool_->allocate(n * sizeof(T)));
    }
    
    void deallocate(T* ptr, size_t n) {
        pool_->deallocate(ptr);
    }
    
    template<typename U>
    bool operator==(const PoolAllocator<U>& other) const {
        return pool_ == other.pool_;
    }
    
    template<typename U>
    bool operator!=(const PoolAllocator<U>& other) const {
        return !(*this == other);
    }
};

template<typename T>
using PoolVector = std::vector<T, PoolAllocator<T>>;

class FinancialMatrix {
private:
    double* data_;
    size_t rows_;
    size_t cols_;
    MemoryPool* pool_;
    
public:
    FinancialMatrix(size_t rows, size_t cols, MemoryPool* pool) 
        : rows_(rows), cols_(cols), pool_(pool) {
        data_ = static_cast<double*>(pool_->allocate(rows * cols * sizeof(double)));
        std::fill(data_, data_ + rows * cols, 0.0);
    }
    
    ~FinancialMatrix() {
        if (data_) {
            pool_->deallocate(data_);
        }
    }
    
    FinancialMatrix(FinancialMatrix&& other) noexcept 
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_), pool_(other.pool_) {
        other.data_ = nullptr;
    }
    
    FinancialMatrix& operator=(FinancialMatrix&& other) noexcept {
        if (this != &other) {
            if (data_) pool_->deallocate(data_);
            data_ = other.data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            pool_ = other.pool_;
            other.data_ = nullptr;
        }
        return *this;
    }
    
    FinancialMatrix(const FinancialMatrix&) = delete;
    FinancialMatrix& operator=(const FinancialMatrix&) = delete;
    
    double& operator()(size_t row, size_t col) {
        return data_[row * cols_ + col];
    }
    
    const double& operator()(size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }
    
    double* data() { return data_; }
    const double* data() const { return data_; }
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    void fill_random() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, 1.0);
        
        for (size_t i = 0; i < rows_ * cols_; i++) {
            data_[i] = dis(gen);
        }
    }
    
    FinancialMatrix multiply(const FinancialMatrix& other) const {
        assert(cols_ == other.rows_);
        
        FinancialMatrix result(rows_, other.cols_, pool_);
        
        const size_t BLOCK_SIZE = 64;
        
        for (size_t i = 0; i < rows_; i += BLOCK_SIZE) {
            for (size_t j = 0; j < other.cols_; j += BLOCK_SIZE) {
                for (size_t k = 0; k < cols_; k += BLOCK_SIZE) {
                    size_t i_max = std::min(i + BLOCK_SIZE, rows_);
                    size_t j_max = std::min(j + BLOCK_SIZE, other.cols_);
                    size_t k_max = std::min(k + BLOCK_SIZE, cols_);
                    
                    for (size_t ii = i; ii < i_max; ii++) {
                        for (size_t jj = j; jj < j_max; jj++) {
                            double sum = 0.0;
                            for (size_t kk = k; kk < k_max; kk++) {
                                sum += data_[ii * cols_ + kk] * other.data_[kk * other.cols_ + jj];
                            }
                            result(ii, jj) += sum;
                        }
                    }
                }
            }
        }
        
        return result;
    }
};

class HighFreqMonteCarloEngine {
private:
    MemoryPool* pool_;
    size_t num_paths_;
    size_t num_steps_;
    
public:
    HighFreqMonteCarloEngine(MemoryPool* pool, size_t paths, size_t steps) 
        : pool_(pool), num_paths_(paths), num_steps_(steps) {}
    
    std::vector<double> simulate_portfolio(const std::vector<double>& initial_prices,
                                          const std::vector<double>& volatilities,
                                          const std::vector<double>& correlations,
                                          double risk_free_rate,
                                          double time_horizon) {
        size_t num_assets = initial_prices.size();
        
        double* paths_data = static_cast<double*>(
            pool_->allocate(num_paths_ * num_steps_ * num_assets * sizeof(double)));
        
        double* random_normals = static_cast<double*>(
            pool_->allocate(num_paths_ * num_steps_ * num_assets * sizeof(double)));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, 1.0);
        
        for (size_t i = 0; i < num_paths_ * num_steps_ * num_assets; i++) {
            random_normals[i] = dis(gen);
        }
        
        double dt = time_horizon / num_steps_;
        std::vector<double> final_prices(num_paths_);
        
        for (size_t path = 0; path < num_paths_; path++) {
            for (size_t asset = 0; asset < num_assets; asset++) {
                paths_data[path * num_steps_ * num_assets + asset] = initial_prices[asset];
            }
            
            for (size_t step = 1; step < num_steps_; step++) {
                for (size_t asset = 0; asset < num_assets; asset++) {
                    size_t idx = path * num_steps_ * num_assets + step * num_assets + asset;
                    size_t prev_idx = path * num_steps_ * num_assets + (step - 1) * num_assets + asset;
                    
                    double drift = (risk_free_rate - 0.5 * volatilities[asset] * volatilities[asset]) * dt;
                    double diffusion = volatilities[asset] * sqrt(dt) * random_normals[idx];
                    
                    paths_data[idx] = paths_data[prev_idx] * exp(drift + diffusion);
                }
            }
            
            double portfolio_value = 0.0;
            for (size_t asset = 0; asset < num_assets; asset++) {
                size_t final_idx = path * num_steps_ * num_assets + (num_steps_ - 1) * num_assets + asset;
                portfolio_value += paths_data[final_idx];
            }
            final_prices[path] = portfolio_value / num_assets;
        }
        
        pool_->deallocate(paths_data);
        pool_->deallocate(random_normals);
        
        return final_prices;
    }
};

class StandardAllocator {
private:
    size_t total_allocated_;
    size_t allocation_count_;
    
public:
    StandardAllocator() : total_allocated_(0), allocation_count_(0) {}
    
    void* allocate(size_t size) {
        if (size == 0) return nullptr;
        
        size_t aligned_size = (size + 31) & ~31;
        void* ptr = aligned_alloc(32, aligned_size);
        
        if (ptr) {
            total_allocated_ += aligned_size;
            allocation_count_++;
        }
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (ptr) {
            aligned_free(ptr);
        }
    }
    
    size_t get_total_allocated() const { return total_allocated_; }
    size_t get_allocation_count() const { return allocation_count_; }
    
    void reset() {
        total_allocated_ = 0;
        allocation_count_ = 0;
    }
};

void benchmark_memory_pool() {
    std::cout << "\n=== Memory Pool vs Standard Allocation Benchmark ===\n\n";
    
    std::vector<size_t> allocation_counts = {1000, 10000, 100000};
    std::vector<size_t> allocation_sizes = {64, 256, 1024, 4096};
    
    for (size_t size : allocation_sizes) {
        std::cout << "Allocation Size: " << size << " bytes\n";
        std::cout << "Count\tPool Time (μs)\tStandard Time (μs)\tSpeedup\tPool Efficiency\n\n";
        
        for (size_t count : allocation_counts) {
            MemoryPool pool;
            std::vector<void*> pool_ptrs(count);
            
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < count; ++i) {
                pool_ptrs[i] = pool.allocate(size);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto pool_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            StandardAllocator std_alloc;
            std::vector<void*> std_ptrs(count);
            
            start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < count; ++i) {
                std_ptrs[i] = std_alloc.allocate(size);
            }
            end = std::chrono::high_resolution_clock::now();
            auto std_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            for (void* ptr : std_ptrs) {
                std_alloc.deallocate(ptr);
            }
            
            double speedup = pool_time > 0 ? (double)std_time / pool_time : 0.0;
            double efficiency = pool.get_efficiency() * 100.0;
            
            std::cout << count << "\t" << pool_time << "\t\t" << std_time << "\t\t\t" 
                      << std::fixed << std::setprecision(2) << speedup << "x\t" 
                      << efficiency << "%\n";
        }
        std::cout << "\n";
    }
}

void benchmark_allocation_deallocation_patterns() {
    std::cout << "=== Allocation/Deallocation Pattern Benchmark ===\n\n";
    
    const size_t num_operations = 10000;
    const size_t allocation_size = 1024;
    
    std::cout << "Test 1: Sequential Allocation/Deallocation\n";
    {
        MemoryPool pool;
        StandardAllocator std_alloc;
        
        std::vector<void*> ptrs(num_operations);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_operations; ++i) {
            ptrs[i] = pool.allocate(allocation_size);
        }
        for (size_t i = 0; i < num_operations; ++i) {
            pool.deallocate(ptrs[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto pool_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_operations; ++i) {
            ptrs[i] = std_alloc.allocate(allocation_size);
        }
        for (size_t i = 0; i < num_operations; ++i) {
            std_alloc.deallocate(ptrs[i]);
        }
        end = std::chrono::high_resolution_clock::now();
        auto std_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout << "  Memory Pool: " << pool_time << " μs\n";
        std::cout << "  Standard: " << std_time << " μs\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
                  << (double)std_time / pool_time << "x\n\n";
    }
    
    std::cout << "Test 2: Random Allocation/Deallocation\n";
    {
        MemoryPool pool;
        StandardAllocator std_alloc;
        
        std::vector<void*> active_ptrs;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 1);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_operations; ++i) {
            if (active_ptrs.empty() || dis(gen)) {
                void* ptr = pool.allocate(allocation_size);
                active_ptrs.push_back(ptr);
            } else {
                size_t idx = gen() % active_ptrs.size();
                pool.deallocate(active_ptrs[idx]);
                active_ptrs.erase(active_ptrs.begin() + idx);
            }
        }
        for (void* ptr : active_ptrs) {
            pool.deallocate(ptr);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto pool_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        active_ptrs.clear();
        gen.seed(rd());
        
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_operations; ++i) {
            if (active_ptrs.empty() || dis(gen)) {
                void* ptr = std_alloc.allocate(allocation_size);
                active_ptrs.push_back(ptr);
            } else {
                size_t idx = gen() % active_ptrs.size();
                std_alloc.deallocate(active_ptrs[idx]);
                active_ptrs.erase(active_ptrs.begin() + idx);
            }
        }
        for (void* ptr : active_ptrs) {
            std_alloc.deallocate(ptr);
        }
        end = std::chrono::high_resolution_clock::now();
        auto std_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout << "  Memory Pool: " << pool_time << " μs\n";
        std::cout << "  Standard: " << std_time << " μs\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
                  << (double)std_time / pool_time << "x\n\n";
    }
}

void benchmark_financial_workloads() {
    std::cout << "=== Financial Workload Benchmark ===\n\n";
    
    std::cout << "Matrix Operations Benchmark:\n";
    std::vector<int> matrix_sizes = {50, 100, 200, 500};
    
    for (int size : matrix_sizes) {
        std::cout << "Matrix size: " << size << "x" << size << "\n";
        
        MemoryPool pool;
        auto start = std::chrono::high_resolution_clock::now();
        
        FinancialMatrix A(size, size, &pool);
        FinancialMatrix B(size, size, &pool);
        
        A.fill_random();
        B.fill_random();
        
        auto C = A.multiply(B);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto pool_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        
        size_t matrix_bytes = size * size * sizeof(double);
        double* std_A = static_cast<double*>(aligned_alloc(32, matrix_bytes));
        double* std_B = static_cast<double*>(aligned_alloc(32, matrix_bytes));
        double* std_C = static_cast<double*>(aligned_alloc(32, matrix_bytes));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, 1.0);
        
        for (int i = 0; i < size * size; ++i) {
            std_A[i] = dis(gen);
            std_B[i] = dis(gen);
            std_C[i] = 0.0;
        }
        
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                for (int k = 0; k < size; ++k) {
                    std_C[i * size + j] += std_A[i * size + k] * std_B[k * size + j];
                }
            }
        }
        
        aligned_free(std_A);
        aligned_free(std_B);
        aligned_free(std_C);
        
        end = std::chrono::high_resolution_clock::now();
        auto std_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "  Memory Pool: " << pool_time << " ms\n";
        std::cout << "  Standard: " << std_time << " ms\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
                  << (double)std_time / pool_time << "x\n";
        std::cout << "  Pool efficiency: " << pool.get_efficiency() * 100.0 << "%\n";
        std::cout << "  Pool allocations: " << pool.get_allocation_count() << "\n\n";
    }
}

void benchmark_original_memory_pool() {
    std::cout << "=== Original Memory Pool Benchmark ===\n\n";
    
    const size_t NUM_ALLOCATIONS = 10000;
    const size_t ALLOCATION_SIZE = 1024;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<void*> std_ptrs;
    for (size_t i = 0; i < NUM_ALLOCATIONS; i++) {
        std_ptrs.push_back(aligned_alloc(32, ALLOCATION_SIZE));
    }
    
    for (void* ptr : std_ptrs) {
        aligned_free(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto std_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    
    MemoryPool pool;
    std::vector<void*> pool_ptrs;
    for (size_t i = 0; i < NUM_ALLOCATIONS; i++) {
        pool_ptrs.push_back(pool.allocate(ALLOCATION_SIZE));
    }
    
    for (void* ptr : pool_ptrs) {
        pool.deallocate(ptr);
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto pool_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Standard allocation: " << std_time.count() << " μs\n";
    std::cout << "Memory pool: " << pool_time.count() << " μs\n";
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) 
              << (double)std_time.count() / pool_time.count() << "x\n\n";
}

void benchmark_memory_allocation() {
    const int allocation_counts[] = {1000, 10000, 100000};
    const int allocation_sizes[] = {64, 256, 1024, 4096};
    
    std::cout << "\n=== Detailed Memory Pool Allocation Benchmark ===\n";
    
    for (int size_idx = 0; size_idx < 4; ++size_idx) {
        int alloc_size = allocation_sizes[size_idx];
        std::cout << "\nAllocation Size: " << alloc_size << " bytes\n";
        std::cout << "Count\tPool Time (μs)\tStandard Time (μs)\tSpeedup\n";
        
        for (int count_idx = 0; count_idx < 3; ++count_idx) {
            int count = allocation_counts[count_idx];
            
            MemoryPool pool;
            
            std::vector<void*> pool_ptrs(count);
            std::vector<void*> std_ptrs(count);
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < count; ++i) {
                pool_ptrs[i] = pool.allocate(alloc_size);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto pool_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < count; ++i) {
                std_ptrs[i] = aligned_alloc(32, alloc_size);
            }
            end = std::chrono::high_resolution_clock::now();
            auto std_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            for (int i = 0; i < count; ++i) {
                aligned_free(std_ptrs[i]);
            }
            
            double speedup = pool_time > 0 ? (double)std_time / pool_time : 0.0;
            std::cout << count << "\t" << pool_time << "\t\t" << std_time << "\t\t" 
                      << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
    }
}

void benchmark_memory_usage_pattern() {
    std::cout << "\n=== Memory Usage Pattern Benchmark ===\n";
    
    const int num_paths = 10000;
    const int path_length = 252;
    const int trials = 100;
    
    MemoryPool pool;
    
    std::cout << "Monte Carlo Memory Pattern (10k paths, 252 steps, 100 trials):\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int trial = 0; trial < trials; ++trial) {
        float* paths = (float*)pool.allocate(num_paths * path_length * sizeof(float));
        
        for (int i = 0; i < num_paths * path_length; ++i) {
            paths[i] = i * 0.001f;
        }
        
        pool.reset();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto pool_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (int trial = 0; trial < trials; ++trial) {
        float* paths = (float*)aligned_alloc(32, num_paths * path_length * sizeof(float));
        
        for (int i = 0; i < num_paths * path_length; ++i) {
            paths[i] = i * 0.001f;
        }
        
        aligned_free(paths);
    }
    end = std::chrono::high_resolution_clock::now();
    auto std_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Pool time: " << pool_time << " μs\n";
    std::cout << "Standard time: " << std_time << " μs\n";
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) 
              << (double)std_time / pool_time << "x\n";
}

void benchmark_memory_pool_advantages() {
    std::cout << "\n=== Memory Pool Advantage Scenarios ===\n\n";
    
    std::cout << "Test 1: Frequent Small Allocations with Reuse\n";
    const size_t num_iterations = 1000;
    const size_t alloc_size = 256;
    
    {
        MemoryPool pool;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t iter = 0; iter < num_iterations; ++iter) {
            std::vector<void*> ptrs;
            for (size_t i = 0; i < 100; ++i) {
                ptrs.push_back(pool.allocate(alloc_size));
            }
            for (void* ptr : ptrs) {
                pool.deallocate(ptr);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto pool_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        
        for (size_t iter = 0; iter < num_iterations; ++iter) {
            std::vector<void*> ptrs;
            for (size_t i = 0; i < 100; ++i) {
                ptrs.push_back(aligned_alloc(32, alloc_size));
            }
            for (void* ptr : ptrs) {
                aligned_free(ptr);
            }
        }
        
        end = std::chrono::high_resolution_clock::now();
        auto std_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout << "  Memory Pool: " << pool_time << " μs\n";
        std::cout << "  Standard: " << std_time << " μs\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
                  << (double)std_time / pool_time << "x\n";
        std::cout << "  Pool Efficiency: " << pool.get_efficiency() * 100.0 << "%\n\n";
    }
    
    std::cout << "Test 2: Memory-Intensive Financial Simulation\n";
    const size_t num_simulations = 100;
    const size_t paths_per_sim = 1000;
    const size_t path_length = 252;
    
    {
        MemoryPool pool;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t sim = 0; sim < num_simulations; ++sim) {
            double* paths = static_cast<double*>(pool.allocate(paths_per_sim * path_length * sizeof(double)));
            
            for (size_t i = 0; i < paths_per_sim * path_length; ++i) {
                paths[i] = sin(i * 0.01) * cos(i * 0.02);
            }
            
            pool.reset();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto pool_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        
        for (size_t sim = 0; sim < num_simulations; ++sim) {
            double* paths = static_cast<double*>(aligned_alloc(32, paths_per_sim * path_length * sizeof(double)));
            
            for (size_t i = 0; i < paths_per_sim * path_length; ++i) {
                paths[i] = sin(i * 0.01) * cos(i * 0.02);
            }
            
            aligned_free(paths);
        }
        
        end = std::chrono::high_resolution_clock::now();
        auto std_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout << "  Memory Pool: " << pool_time << " μs\n";
        std::cout << "  Standard: " << std_time << " μs\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
                  << (double)std_time / pool_time << "x\n";
        std::cout << "  Pool Efficiency: " << pool.get_efficiency() * 100.0 << "%\n\n";
    }
}

int main() {
    std::cout << "=== Memory Pool Performance Analysis ===\n";
    
    benchmark_memory_pool_advantages();
    benchmark_original_memory_pool();
    benchmark_allocation_deallocation_patterns();
    benchmark_financial_workloads();
    
    std::cout << "\n=== High-Frequency Monte Carlo Simulation ===\n";
    
    MemoryPool simulation_pool;
    HighFreqMonteCarloEngine engine(&simulation_pool, 50000, 252);  
    
    std::vector<double> initial_prices = {100.0, 105.0, 98.0, 102.0};
    std::vector<double> volatilities = {0.2, 0.25, 0.18, 0.22};
    std::vector<double> correlations = {1.0, 0.3, 0.2, 0.1};  
    
    auto start = std::chrono::high_resolution_clock::now();
    auto final_prices = engine.simulate_portfolio(initial_prices, volatilities, correlations, 0.05, 1.0);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto simulation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double mean = 0.0, variance = 0.0;
    for (double price : final_prices) {
        mean += price;
    }
    mean /= final_prices.size();
    
    for (double price : final_prices) {
        variance += (price - mean) * (price - mean);
    }
    variance /= (final_prices.size() - 1);
    
    std::cout << "Simulation completed in " << simulation_time.count() << " ms\n";
    std::cout << "Mean portfolio value: " << std::fixed << std::setprecision(2) << mean << "\n";
    std::cout << "Portfolio volatility: " << std::fixed << std::setprecision(4) << sqrt(variance) << "\n";
    std::cout << "Pool efficiency: " << std::fixed << std::setprecision(2) 
              << simulation_pool.get_efficiency() * 100.0 << "%\n";
    std::cout << "Pool allocations: " << simulation_pool.get_allocation_count() << "\n";
    
    std::cout << "\nMemory Pool Final Statistics:\n";
    std::cout << "Total allocated: " << simulation_pool.get_total_allocated() << " bytes\n";
    std::cout << "Total used: " << simulation_pool.get_total_used() << " bytes\n";
    std::cout << "Allocation count: " << simulation_pool.get_allocation_count() << "\n";
    std::cout << "Efficiency: " << simulation_pool.get_efficiency() * 100.0 << "%\n";
    
    return 0;
}
