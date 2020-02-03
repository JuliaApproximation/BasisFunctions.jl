
# In this file we define the interface for a number of generic functions:
# See the individual files for details on the interfaces.

include("transform.jl")
include("evaluation.jl")
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


for op in (:interpolation, :leastsquares)
    @eval $op(::Type{T}, s1::TensorProductDict, s2::TensorProductDict; options...) where {T} =
        tensorproduct(map( (u,v) -> $op(T, u, v; options...), elements(s1), elements(s2))...)
end

dense_evaluation(::Type{T}, s1::TensorProductDict, s2::TensorProductDict; options...) where {T} =
    tensorproduct(map( (u,v) -> dense_evaluation(T, u, v; options...), elements(s1), elements(s2))...)

for op in (:approximation, )
    @eval $op(s::TensorProductDict; options...) =
        tensorproduct(map( u -> $op(u; options...), elements(s))...)
end
