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

export test_generic_grid, test_interval_grid
export grid_iterator1, grid_iterator2
include("test_grids.jl")

export test_generic_operator_interface
include("test_operator.jl")

export test_orthogonality_orthonormality
include("test_functionality.jl")
end
