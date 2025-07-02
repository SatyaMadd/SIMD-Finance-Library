# SIMD Finance

A high-performance C++ library for financial computations using SIMD (Single Instruction, Multiple Data) optimizations. This library provides accelerated implementations of common financial algorithms including option pricing, Monte Carlo simulations, and portfolio optimization.

## Features

- **Black-Scholes Option Pricing**: SIMD-optimized Black-Scholes model for European options
- **Monte Carlo Simulations**: Vectorized Monte Carlo methods for option pricing and risk analysis
- **Portfolio Optimization**: SIMD-accelerated portfolio optimization algorithms
- **Matrix Operations**: Fast matrix multiplication and correlation calculations
- **Memory Management**: Custom memory pool for efficient memory allocation
- **Random Number Generation**: SIMD-optimized random number generation

## Architecture Support

The library is designed to work on multiple architectures:
- **ARM64** (Apple Silicon, ARM processors) - uses ARM NEON intrinsics
- **x86_64** (Intel/AMD processors) - uses AVX/AVX2 intrinsics

## Components

### Core Modules

- `black_scholes_simd.cpp` - SIMD-optimized Black-Scholes option pricing
- `monte_carlo.cpp` - Monte Carlo simulation engine
- `portfolio_optimization.cpp` - Portfolio optimization algorithms
- `correlation_engine.cpp` - Correlation matrix computations
- `matrix_multiply.cpp` - SIMD matrix operations
- `array_multiply.cpp` - Vectorized array operations
- `memory_pool.cpp` - Custom memory allocator
- `simd_random.cpp` - SIMD random number generation

## Requirements

- C++17 compatible compiler
- GCC 7+ or Clang 5+ or MSVC 2017+
- CPU with SIMD support (ARM NEON or x86 AVX)

## Building

```bash
git clone <repository-url>
cd simd_finance

mkdir build
cd build

g++ -std=c++17 -O3 -march=native -fopenmp src/*.cpp -o simd_finance
```

## Usage Example

```cpp
#include "black_scholes_simd.cpp"
#include "monte_carlo.cpp"

int main() {
    SIMDBlackScholes bs;
    double call_price = bs.calculate_call_price(100.0, 105.0, 0.25, 0.05, 0.2);
    
    MonteCarloSimulator mc(1000000, 252, 0.05, 0.2, 100.0);
    double option_price = mc.price_european_call(105.0, 0.25);
    
    return 0;
}
```

## Performance Features

- **Vectorization**: Utilizes SIMD instructions for parallel computation
- **Memory Optimization**: Custom memory pools reduce allocation overhead
- **Cache Efficiency**: Data structures optimized for cache performance
- **Multi-threading**: OpenMP support for parallel execution


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


