
"Wrap an operator and return an expansion rather than coefficients."
struct FunOperator
    op
end

function (*)(op::FunOperator, args...)
    coef = (*)(op.op, args...)
    Expansion(dest(op.op), coef)
end



#####################################
# Operators for tensor product dictionaries
#####################################

# We make TensorProductOperator's for each generic operator, when invoked with
# TensorProductDict's.


transform_from_grid(T, s1::GridBasis, s2::TensorProductDict, grid::ProductGrid; options...) =
    tensorproduct(map( (u,v,w) -> transform_from_grid(T, u,v,w; options...), components(s1), components(s2), components(grid))...)

transform_to_grid(T, s1::TensorProductDict, s2::GridBasis, grid::ProductGrid; options...) =
    tensorproduct(map( (u,v,w) -> transform_to_grid(T, u,v,w; options...), components(s1), components(s2), components(grid))...)

for op in (:extension, :restriction, :conversion)
    @eval $op(::Type{T}, src::TensorProductDict, dest::TensorProductDict; options...) where {T} =
        tensorproduct(map( (u,v) -> $op(T, u, v; options...), components(src), components(dest))...)
end


for op in (:interpolation, :leastsquares)
    @eval $op(::Type{T}, s1::TensorProductDict, s2::TensorProductDict; options...) where {T} =
        tensorproduct(map( (u,v) -> $op(T, u, v; options...), components(s1), components(s2))...)
end

default_evaluation(::Type{T}, s1::TensorProductDict, s2::TensorProductDict; options...) where {T} =
    tensorproduct(map( (u,v) -> default_evaluation(T, u, v; options...), components(s1), components(s2))...)

for op in (:approximation, )
    @eval $op(s::TensorProductDict; options...) =
        tensorproduct(map( u -> $op(u; options...), components(s))...)
end
