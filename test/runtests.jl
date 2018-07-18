using BasisFunctions

if VERSION < v"0.7-"
    using Base.Test
else
    using Test, LinearAlgebra
end

# write your own tests here

include("test_suite.jl")
