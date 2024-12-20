# BasisFunctions.jl

*A flexible and extensible Julia package for function approximation using various basis functions.*

| **Build Status** | **Coverage** |
|------------------|--------------|
| [![Build Status](https://github.com/JuliaApproximation/BasisFunctions.jl/workflows/CI/badge.svg)](https://github.com/JuliaApproximation/BasisFunctions.jl/actions) | [![Coverage](https://codecov.io/gh/JuliaApproximation/BasisFunctions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaApproximation/BasisFunctions.jl)

This package provides a framework for a number of standard basis functions to perform function approximation. The most developed examples are Chebyshev polynomials and Fourier series. This package was developed mainly for use in the package FrameFun, which centers around the numerical approximation of functions using approximation-theoretical frames. It can be used separately as well, yet by far this code is not fully featured and it is not intended to be so. For more complete software packages to manipulate numerical function approximations, please consider [Chebfun](http://www.chebfun.org) (in Matlab) or [ApproxFun](https://github.com/JuliaApproximation/ApproxFun.jl) (in Julia).

The main goal of the package is generality: one can select a subset of a basis, combine several bases or create tensor products. Features of a basis are available via introspection: one can ask a basis what its differentiation operator is, how to use it to approximate a function, whether there is an associated transform and so on. The flexibility is meant to help explore novel algorithms in function approximation research.

Operations on functions are implemented via linear operators that act on the coefficients of an expansion. The focus of the package lies on finite expansions, i.e. with some fixed length N. Care is being taken in the implementation of operators such that operators do not allocate memory (or at most an N-independent amount) when they are applied, only when they are created.

Some examples of how to use this code can be found in the [FrameFun](https://github.com/JuliaApproximation/FrameFun.jl) package.



## Installation

The package can be installed with the package manager. You typically want to combine the BasisFunctions package with DomainSets. From the Julia REPL, type `]` to enter Pkg mode and run

```julia
pkg> add DomainSets, BasisFunctions
```
