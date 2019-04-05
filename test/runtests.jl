module test_suite

using BasisFunctions, BasisFunctions.Test, DomainSets, StaticArrays

import BasisFunctions.Test:
    supports_approximation,
    supports_interpolation,
    suitable_function,
    suitable_interpolation_grid

using Test, Random, FFTW, LinearAlgebra

linspace(a,b,c) = range(a, stop=b, length=c)

my_rand = rand
Random.seed!(1234)

BF = BasisFunctions
const show_timings = false

##########
# Testing
##########

Delimit("Utilities")
include("util/test_utilities.jl")

Delimit("Grids")
include("test_grids.jl")

Delimit("Operators")
include("test_operators.jl")

Delimit("Dictionaries")
include("dictionaries/test_dictionaries.jl")

Delimit("Check evaluations, interpolations, extensions, expansions")
include("test_fourier.jl")
include("test_chebyshev.jl")
include("test_ops.jl")

Delimit("Generic OPS")
include("test_generic_OPS.jl")

Delimit("Plots")
include("test_plots.jl")

println(" All tests passed!")

end # module
