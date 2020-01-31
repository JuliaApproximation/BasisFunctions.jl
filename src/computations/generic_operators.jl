
# In this file we define the interface for the following generic functions:
#
# Approximation:
# - interpolation_operator
# - approximation_operator
# - leastsquares_operator
# - transform_operator
#
# Calculus:
# - differentiation_operator
# - antidifferentation_operator
#
# These operators are also defined for TensorProductDict's.
#
# See the individual files for details on the interfaces.

include("transform.jl")

include("evaluation.jl")

include("interpolation.jl")

include("leastsquares.jl")

include("approximation.jl")

include("differentiation.jl")



#####################################
# Operators for tensor product dictionaries
#####################################

# We make TensorProductOperator's for each generic operator, when invoked with
# TensorProductDict's.


transform_from_grid(T, s1::GridBasis, s2::TensorProductDict, grid::ProductGrid; options...) =
    tensorproduct(map( (u,v,w) -> transform_from_grid(T, u,v,w; options...), elements(s1), elements(s2), elements(grid))...)

transform_to_grid(T, s1::TensorProductDict, s2::GridBasis, grid::ProductGrid; options...) =
    tensorproduct(map( (u,v,w) -> transform_to_grid(T, u,v,w; options...), elements(s1), elements(s2), elements(grid))...)

for op in (:extension, :restriction, :conversion)
    @eval $op(::Type{T}, src::TensorProductDict, dest::TensorProductDict; options...) where {T} =
        tensorproduct(map( (u,v) -> $op(T, u, v; options...), elements(src), elements(dest))...)
end


for op in (:interpolation_operator, :leastsquares_operator)
    @eval $op(s1::TensorProductDict, s2::TensorProductDict; options...) =
        tensorproduct(map( (u,v) -> $op(u, v; options...), elements(s1), elements(s2))...)
end

dense_evaluation(s1::TensorProductDict, s2::TensorProductDict; options...) =
    tensorproduct(map( (u,v) -> dense_evaluation(u, v; options...), elements(s1), elements(s2))...)

for op in (:approximation_operator, )
    @eval $op(s::TensorProductDict; options...) =
        tensorproduct(map( u -> $op(u; options...), elements(s))...)
end

for op in (:differentiation_operator, :antidifferentiation_operator)
    @eval function $op(s1::TensorProductDict, s2::TensorProductDict, order::NTuple; options...)
        @assert length(order) == dimension(s1)
        @assert length(order) == dimension(s2)
        tensorproduct(map( (u,v,w) -> $op(u, v, w; options...), elements(s1), elements(s2), order)...)
    end
end
