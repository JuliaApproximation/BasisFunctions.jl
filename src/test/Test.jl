module Test

using ..BasisFunctions, DomainSets, StaticArrays
using FillArrays: Ones, Eye
import ..BasisFunctions: instantiate
using Test, LinearAlgebra
BF = BasisFunctions

export Delimit, delimit, instantiate
export point_in_domain, point_outside_domain, random_point_in_domain, fixed_point_in_domain
export random_index
export test_tolerance
include("test_util_functions.jl")

export test_generic_dict_interface
export supports_approximation, supports_interpolation
export suitable_interpolation_grid, suitable_function
include("test_dictionary.jl")

export test_generic_operator_interface
include("test_operator.jl")

export test_orthogonality_orthonormality
include("test_functionality.jl")

export generic_test_discrete_measure, generic_test_measure
function generic_test_discrete_measure(measure)
    io = IOBuffer()
    show(io, measure)
    @test length(take!(io))>0
    @test length(weights(measure)) == length(grid(measure))

end

function generic_test_measure(measure)
    io = IOBuffer()
    show(io, measure)
    @test length(take!(io))>0
    support(measure)
    x = rand()
    @test weight(measure,x) ≈ weightfunction(measure)(x)
end
end
