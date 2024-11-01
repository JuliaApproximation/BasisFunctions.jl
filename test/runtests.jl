module test_suite

using Test,
    BasisFunctions, BasisFunctions.Test

using DoubleFloats,
    FFTW,
    GenericFFT,
    LinearAlgebra,
    Random,
    StaticArrays,
    DomainSets,
    DomainIntegrals

import BasisFunctions.Test:
    supports_approximation,
    supports_interpolation,
    suitable_function,
    suitable_interpolation_grid

⊕ = BasisFunctions.:⊕

my_rand = rand
Random.seed!(1234)

BF = BasisFunctions
const show_timings = false

# const LargeFloat = Double64
const LargeFloat = BigFloat

##########
# Testing
##########
ENV["JULIA_DEBUG"]="all"

Delimit("Utilities")
include("util/test_utilities.jl")
include("util/test_pgfplots.jl")

Delimit("Operators")
include("test_operators.jl")

Delimit("Dictionaries")
include("dictionaries/test_dictionaries.jl")
include("dictionaries/test_pwconstants.jl")

Delimit("Check evaluations, interpolations, extensions, expansions")
include("test_fourier.jl")
include("test_chebyshev.jl")
include("test_ops.jl")
include("test_gram.jl")

Delimit("Generic OPS")
include("test_generic_OPS.jl")

Delimit("Plots")
include("test_plots.jl")

println(" All tests passed!")

end # module
