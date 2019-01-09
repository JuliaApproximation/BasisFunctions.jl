
"The zero operator maps everything to zero."
struct ZeroOperator{T} <: DictionaryOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
end

ZeroOperator(src::Dictionary, dest::Dictionary = src) =
    ZeroOperator{op_eltype(src, dest)}(src, dest)

similar_operator(op::ZeroOperator{T}, src, dest) where {T} =
    ZeroOperator{promote_type(T,op_eltype(src,dest))}(src, dest)

unsafe_wrap_operator(src, dest, op::ZeroOperator) = similar_operator(op, src, dest)

# We can only be in-place if the numbers of coefficients of src and dest match
isinplace(op::ZeroOperator) = length(src(op))==length(dest(op))

isdiagonal(::ZeroOperator) = true

adjoint(op::ZeroOperator) = similar_operator(op, dest(op), src(op))

matrix!(op::ZeroOperator, a) = (fill!(a, 0); a)

function apply_inplace!(op::ZeroOperator, coef_srcdest)
    fill!(coef_srcdest, 0)
    coef_srcdest
end

function apply!(op::ZeroOperator, coef_dest, coef_src)
    fill!(coef_dest, 0)
    coef_dest
end

diagonal(op::ZeroOperator) = zeros(eltype(op), min(length(src(op)), length(dest(op))))

unsafe_diagonal(op::ZeroOperator, i) = zero(eltype(op))

unsafe_getindex(op::ZeroOperator, i, j) = zero(eltype(op))
