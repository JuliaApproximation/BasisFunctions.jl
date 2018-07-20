# test_suite.jl
module test_suite

using BasisFunctions, BasisFunctions.Test, Domains, StaticArrays
import BasisFunctions.Test: supports_approximation, supports_interpolation, suitable_function, suitable_interpolation_grid

if VERSION < v"0.7-"
    using Base.Test
    my_rand(T, a...) = map(T, rand(a...))
    ComplexF64 = Complex128
    ComplexF32 = Complex64
else
    using Test, Random, FFTW, LinearAlgebra
    linspace(a,b,c) = range(a, stop=b, length=c)
    my_rand = rand
end

if VERSION < v"0.7-"
    types = [Float64,BigFloat,]
else
    warn("Postpone BigFloat testing until FastTransforms is fixed. ")
    types = [Float64,]
end

srand(1234)
BF = BasisFunctions
const show_timings = false

######### #
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

Delimit("Gram")
include("test_gram.jl")

Delimit("Check evaluations, interpolations, extensions, setexpansions")
include("test_fourier.jl")
include("test_ops.jl")

Delimit("Generic OPS")
include("test_generic_OPS.jl")

Delimit("DCTI")
include("test_DCTI.jl")

Delimit("Plots")
include("test_plots.jl")

println(" All tests passed!")

end # module
