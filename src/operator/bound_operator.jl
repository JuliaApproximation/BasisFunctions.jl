# bound_operator.jl

"""
A `BoundOperator` associates specific function spaces with a generic operator
that may only define an action on discrete spaces with variable size.
See also `UnboundOperator` below.
"""
struct BoundOperator{T,O <: AbstractOperator{T}} <: AbstractOperator{T}
    src     ::  Span
    dest    ::  Span
    action  ::  O

    function BoundOperator{T}(src, dest, action) where {T}
        @assert length(src) == length(dest)
        new(src, dest, action)
    end
end

function BoundOperator(::Type{T}, src, dest, action) where {T}
    S, D, A = op_eltypes(src, dest, T)
    BoundOperator{A}(promote_coeftype(src, S), promote_coeftype(dest, D),
        promote_eltype(action, A))
end

similar_operator(op::BoundOperator, ::Type{S}, src, dest) where {S} =
    BoundOperator(S, src, dest, action(op))

action(op::BoundOperator) = op.action

is_bound(op::BoundOperator) = true

for op in (:is_inplace, :is_diagonal)
    @eval $op(bop::BoundOperator) = $op(action(bop))
end

function apply_inplace!(op::BoundOperator, coef_srcdest)
    @assert is_inplace(op)
    apply_inplace!(action(op), coef_srcdest)
end

apply!(op::BoundOperator, coef_dest, coef_src) = apply!(action(op), coef_dest, coef_src)

inv(op::BoundOperator{T}) where {T} =
    BoundOperator(T, dest(op), src(op), inv(action(op)))

ctranspose(op::BoundOperator{T}) where {T} =
    BoundOperator(T, dest(op), src(op), ctranspose(action(op)))

matrix!(op::BoundOperator, a) = matrix!(action(op), a)

simplify(op::BoundOperator) = op



"""
An `UnboundOperator` defines an action that can be appied to many kinds of
coefficients. Hence, it does not specify which coefficients it expects and
applies the action adaptively depending on the input.
"""
abstract type UnboundOperator{T} <: AbstractOperator{T}
end

is_bound(op::UnboundOperator) = false
