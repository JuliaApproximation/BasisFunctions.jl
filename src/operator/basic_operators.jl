# basic_operators.jl

"""
The identity operator between two (possibly different) function sets.
"""
struct IdentityOperator{T} <: DictionaryOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary

    function IdentityOperator{T}(src, dest) where {T}
        @assert length(src) == length(dest)
        new(src, dest)
    end
end

IdentityOperator(src::Dictionary, dest::Dictionary = src) =
    IdentityOperator{op_eltype(src, dest)}(src, dest)

similar_operator(::IdentityOperator, src, dest) = IdentityOperator(src, dest)

unsafe_wrap_operator(src, dest, op::IdentityOperator) = IdentityOperator(src, dest)

@add_properties(IdentityOperator, is_inplace, is_diagonal)

inv(op::IdentityOperator) = IdentityOperator(dest(op), src(op))

adjoint(op::IdentityOperator)::DictionaryOperator = IdentityOperator(dest(op), src(op))

function matrix!(op::IdentityOperator, a)
    @assert size(a,1) == size(a,2)

    a[:] .= zero(eltype(op))
    for i in 1:size(a,1)
        a[i,i] = one(eltype(op))
    end
    a
end

diagonal(op::IdentityOperator) = ones(eltype(op), length(src(op)))

unsafe_diagonal(op::IdentityOperator, i) = one(eltype(op))

apply_inplace!(op::IdentityOperator, coef_srcdest) = coef_srcdest

string(op::IdentityOperator) = "Identity Operator"

"A ScalingOperator represents multiplication by a scalar."
struct ScalingOperator{T} <: DictionaryOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    scalar  ::  T

    function ScalingOperator{T}(src, dest, scalar) where T
        @assert length(src) == length(dest)
        new(src, dest, scalar)
    end
end

ScalingOperator(src::Dictionary, scalar::Number) = ScalingOperator(src, src, scalar)

ScalingOperator(src::Dictionary, dest::Dictionary, scalar) =
    ScalingOperator{op_eltype(src,dest)}(src, dest, scalar)

scalar(op::ScalingOperator) = op.scalar

similar_operator(op::ScalingOperator, src, dest) = ScalingOperator(src, dest, scalar(op))

unsafe_wrap_operator(src, dest, op::ScalingOperator) = similar_operator(op, src, dest)

@add_properties(ScalingOperator, is_inplace, is_diagonal)

adjoint(op::ScalingOperator)::DictionaryOperator = ScalingOperator(dest(op), src(op), conj(scalar(op)))


inv(op::ScalingOperator) = ScalingOperator(dest(op), src(op), inv(scalar(op)))

function apply_inplace!(op::ScalingOperator, coef_srcdest)
    for i in eachindex(coef_srcdest)
        coef_srcdest[i] *= op.scalar
    end
    coef_srcdest
end

# Extra definition for out-of-place version to avoid making an intermediate copy
function apply!(op::ScalingOperator, coef_dest, coef_src)
    for i in eachindex(coef_src, coef_dest)
        coef_dest[i] = op.scalar * coef_src[i]
    end
    coef_dest
end

for op in (:+, :*, :-, :/)
    @eval $op(op1::ScalingOperator, op2::ScalingOperator) =
        (@assert size(op1) == size(op2); ScalingOperator(src(op1), dest(op1), $op(scalar(op1), scalar(op2))))
end

diagonal(op::ScalingOperator) = [scalar(op) for i in 1:length(src(op))]

unsafe_diagonal(op::ScalingOperator, i) = scalar(op)

function matrix!(op::ScalingOperator, a)
    a[:] .= 0
    for i in 1:min(size(a,1),size(a,2))
        a[i,i] = scalar(op)
    end
    a
end

unsafe_getindex(op::ScalingOperator{T}, i, j) where {T} = i == j ? convert(T, op.scalar) : convert(T, 0)


# default implementation for scalar multiplication is a scaling operator
*(scalar::Number, op::DictionaryOperator) = ScalingOperator(dest(op), scalar) * op

string(op::ScalingOperator) = "Scaling by $(scalar(op))"

symbol(S::ScalingOperator) = "α"

"The zero operator maps everything to zero."
struct ZeroOperator{T} <: DictionaryOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
end

ZeroOperator(src::Dictionary, dest::Dictionary = src) = ZeroOperator{op_eltype(src, dest)}(src, dest)

similar_operator(op::ZeroOperator, src, dest) = ZeroOperator(src, dest)

unsafe_wrap_operator(src, dest, op::ZeroOperator) = similar_operator(op, src, dest)

# We can only be in-place if the numbers of coefficients of src and dest match
is_inplace(op::ZeroOperator) = length(src(op))==length(dest(op))

is_diagonal(::ZeroOperator) = true

adjoint(op::ZeroOperator)::DictionaryOperator = ZeroOperator(dest(op), src(op))

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



"A diagonal operator is represented by a diagonal matrix."
struct DiagonalOperator{T} <: DictionaryOperator{T}
    src         ::  Dictionary
    dest        ::  Dictionary
    diagonal    ::  Vector{T}    # We store the diagonal in a vector

    function DiagonalOperator{T}(src, dest, diagonal) where T
        @assert length(src) == length(dest)
        new(src, dest, diagonal)
    end
end

DiagonalOperator(src::Dictionary, dest::Dictionary, diagonal::Vector) =
    DiagonalOperator{op_eltype(src,dest)}(src, dest, diagonal)

DiagonalOperator(src::Dictionary, dest::Dictionary, diagonal::AbstractVector) =
    DiagonalOperator(src, dest, collect(diagonal))

DiagonalOperator(src::Dictionary, diagonal::AbstractVector) = DiagonalOperator(src, src, diagonal)

DiagonalOperator(diagonal::AbstractVector{T}) where {T} =
    DiagonalOperator(DiscreteVectorDictionary{T}(length(diagonal)), diagonal)

similar_operator(op::DiagonalOperator, src, dest) = DiagonalOperator(src, dest, diagonal(op))

unsafe_wrap_operator(src, dest, op::DiagonalOperator) = similar_operator(op, src, dest)

@add_properties(DiagonalOperator, is_inplace, is_diagonal)

diagonal(op::DiagonalOperator) = copy(op.diagonal)

inv(op::DiagonalOperator) = DiagonalOperator(dest(op), src(op), inv.(op.diagonal))

adjoint(op::DiagonalOperator)::DictionaryOperator = DiagonalOperator(dest(op), src(op), conj.(diagonal(op)))

function matrix!(op::DiagonalOperator, a)
    a[:] .= 0
    for i in 1:min(size(a,1),size(a,2))
        a[i,i] = op.diagonal[i]
    end
    a
end

function apply_inplace!(op::DiagonalOperator, coef_srcdest)
    for i in 1:length(coef_srcdest)
        coef_srcdest[i] *= op.diagonal[i]
    end
    coef_srcdest
end

# Extra definition for out-of-place version to avoid making an intermediate copy
function apply!(op::DiagonalOperator, coef_dest, coef_src)
    for i in 1:length(coef_dest)
        coef_dest[i] = op.diagonal[i] * coef_src[i]
    end
    coef_dest
end

matrix(op::DiagonalOperator) = Matrix(Diagonal(diagonal(op)))


##################################
# Promotions and simplifications
##################################

## PROMOTION RULES

promote_rule(::Type{ScalingOperator{T}}, ::Type{IdentityOperator{T}}) where T = ScalingOperator{T}
promote_rule(::Type{DiagonalOperator{T}}, ::Type{IdentityOperator{T}}) where T = DiagonalOperator{T}

promote_rule(::Type{ScalingOperator{T}}, ::Type{ScalingOperator{T}}) where T = ScalingOperator{T}
promote_rule(::Type{DiagonalOperator{T}}, ::Type{ScalingOperator{T}}) where T = DiagonalOperator{T}

promote_rule(::Type{ScalingOperator{T}}, ::Type{ZeroOperator{T}}) where T = ScalingOperator{T}
promote_rule(::Type{DiagonalOperator{T}}, ::Type{ZeroOperator{T}}) where T = DiagonalOperator{T}

## CONVERSIONS

convert(::Type{ScalingOperator{T}}, op::IdentityOperator{T}) where T =
    ScalingOperator(src(op), dest(op), one(T))
convert(::Type{DiagonalOperator{T}}, op::IdentityOperator{T}) where T =
    DiagonalOperator(src(op), dest(op), ones(T, length(src(op))))

convert(::Type{DiagonalOperator{T}}, op::ScalingOperator{T}) where T =
    DiagonalOperator(src(op), dest(op), T(scalar(op))*ones(T,length(src(op))))

convert(::Type{DiagonalOperator{T}}, op::ZeroOperator{T}) where T =
    DiagonalOperator(src(op), dest(op), zeros(T, length(src(op))))


## SIMPLIFICATIONS

# The simplify routine is used to simplify the action of composite operators.
# It is called with one or two arguments. The result should be an operator
# (or a tuple of operators in case of two arguments) that does the same thing,
# but may be faster to apply. The correct chaining of src and destination sets
# for consecutive operators is verified before simplification, and does not need
# to be respected afterwards. Only the sizes (and representations) of the coefficients
# must match.

# The identity operator is, well, the identity
simplify(op::IdentityOperator) = nothing
simplify(op1::IdentityOperator, op2::IdentityOperator) = (op1,)
simplify(op1::IdentityOperator, op2::DictionaryOperator) = (op2,)
simplify(op1::DictionaryOperator, op2::IdentityOperator) = (op1,)

(*)(a::Number, op::IdentityOperator) = ScalingOperator(src(op), dest(op), a)
(*)(op::IdentityOperator, a::Number) = a*op

# The zero operator annihilates all other operators
simplify(op1::ZeroOperator, op2::ZeroOperator) = (op1,)
simplify(op1::ZeroOperator, op2::DictionaryOperator) = (op1,)
simplify(op1::DictionaryOperator, op2::ZeroOperator) = (op2,)



(*)(op1::DiagonalOperator, op2::DiagonalOperator) = DiagonalOperator(src(op2), dest(op1), diagonal(op1) .* diagonal(op2))
(*)(op1::ScalingOperator, op2::DiagonalOperator) = DiagonalOperator(src(op2), dest(op1), scalar(op1) * diagonal(op2))
(*)(op2::DiagonalOperator, op1::ScalingOperator) = op1 * op2

(+)(op1::DiagonalOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), diagonal(op1) + diagonal(op2))
(+)(op1::ScalingOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), scalar(op1) +  diagonal(op2))
(+)(op2::DiagonalOperator, op1::ScalingOperator) = op1 + op2
