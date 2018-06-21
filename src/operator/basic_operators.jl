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

IdentityOperator(src::Dictionary, dest::Dictionary = src) = IdentityOperator(op_eltype(src, dest), src, dest)

function IdentityOperator(::Type{T}, src, dest) where {T}
    IdentityOperator{T}(src, dest)
end

similar_operator(::IdentityOperator, ::Type{S}, src, dest) where {S} = IdentityOperator(S, src, dest)

@add_properties(IdentityOperator, is_inplace, is_diagonal)

inv(op::IdentityOperator) = IdentityOperator(eltype(op), dest(op), src(op))

ctranspose(op::IdentityOperator) = IdentityOperator(eltype(op), dest(op), src(op))

function matrix!(op::IdentityOperator, a)
    @assert size(a,1) == size(a,2)

    a[:] = zero(eltype(op))
    for i in 1:size(a,1)
        a[i,i] = one(eltype(op))
    end
    a
end

diagonal(op::IdentityOperator) = ones(eltype(op), length(src(op)))

unsafe_diagonal(op::IdentityOperator, i) = one(eltype(op))

apply_inplace!(op::IdentityOperator, coef_srcdest) = coef_srcdest

string(op::IdentityOperator) = "Identity Operator"

"""
A ScalingOperator is the identity operator up to a scaling.
"""
struct ScalingOperator{T} <: DictionaryOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    scalar  ::  T

    function ScalingOperator{T}(src, dest, scalar) where {T}
        @assert length(src) == length(dest)
        new(src, dest, scalar)
    end
end

ScalingOperator(src::Dictionary, scalar::Number) = ScalingOperator(src, src, scalar)

ScalingOperator(src::Dictionary, dest::Dictionary, scalar) =
    ScalingOperator(promote_type(op_eltype(src,dest),typeof(scalar)), src, dest, scalar)

function ScalingOperator(::Type{T}, src, dest, scalar) where {T}
    ScalingOperator{T}(src, dest, scalar)
end

similar_operator(op::ScalingOperator, ::Type{S}, src, dest) where {S} =
    ScalingOperator(S, src, dest, scalar(op))

scalar(op::ScalingOperator) = op.scalar

is_inplace(::ScalingOperator) = true
is_diagonal(::ScalingOperator) = true

ctranspose(op::ScalingOperator) = ScalingOperator(dest(op), src(op), conj(scalar(op)))

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
    a[:] = 0
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

ZeroOperator(src::Dictionary, dest::Dictionary = src) = ZeroOperator(op_eltype(src, dest), src, dest)

function ZeroOperator(::Type{T}, src::Dictionary, dest::Dictionary) where {T}
    ZeroOperator{T}(src, dest)
end

similar_operator(op::ZeroOperator, ::Type{S}, src, dest) where {S} =
    ZeroOperator(S, src, dest)

# We can only be in-place if the numbers of coefficients of src and dest match
is_inplace(op::ZeroOperator) = length(src(op))==length(dest(op))

is_diagonal(::ZeroOperator) = true

ctranspose(op::ZeroOperator) = ZeroOperator(dest(op), src(op))

matrix!(op::ZeroOperator, a) = (fill!(a, 0); a)

apply_inplace!(op::ZeroOperator, coef_srcdest) = (fill!(coef_srcdest, zero(eltype(op))); coef_srcdest)

apply!(op::ZeroOperator, coef_dest, coef_src) = (fill!(coef_dest, zero(eltype(op))); coef_dest)

diagonal(op::ZeroOperator) = zeros(eltype(op), min(length(src(op)), length(dest(op))))

unsafe_diagonal(op::ZeroOperator, i) = zero(eltype(op))

unsafe_getindex{T}(op::ZeroOperator{T}, i, j) = zero(eltype(op))





"""
A diagonal operator is represented by a diagonal matrix.

Several other operators can be converted into a diagonal matrix, and this
conversion happens automatically when such operators are combined into a composite
operator.
"""
struct DiagonalOperator{T} <: DictionaryOperator{T}
    src         ::  Dictionary
    dest        ::  Dictionary
    # We store the diagonal in a vector
    diagonal    ::  Vector{T}
end

DiagonalOperator(diagonal::AbstractVector{T}) where {T} =
    DiagonalOperator(DiscreteVectorSet{T}(length(diagonal)), diagonal)

DiagonalOperator(src::Dictionary, diagonal::AbstractVector) = DiagonalOperator(eltype(diagonal), src, src, diagonal)

# Intercept the default constructor
DiagonalOperator(src::Dictionary, dest::Dictionary, diagonal::AbstractVector{T}) where {T} = DiagonalOperator(T, src, dest, diagonal)

function DiagonalOperator(::Type{T}, src::Dictionary, dest::Dictionary, diagonal) where {T}
    T1 = promote_type(coeftype(src),coeftype(dest))
    DiagonalOperator{T1}(src, dest, convert(Vector{T1}, diagonal))
end

similar_operator(op::DiagonalOperator, ::Type{S}, src, dest) where {S} =
    DiagonalOperator(S, src, dest, diagonal(op))

diagonal(op::DiagonalOperator) = copy(op.diagonal)

is_inplace(::DiagonalOperator) = true
is_diagonal(::DiagonalOperator) = true

inv(op::DiagonalOperator) = DiagonalOperator(dest(op), src(op), inv.(op.diagonal))

ctranspose(op::DiagonalOperator) = DiagonalOperator(dest(op), src(op), conj.(diagonal(op)))

function matrix!(op::DiagonalOperator, a)
    a[:] = 0
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

matrix(op::DiagonalOperator) = diagm(diagonal(op))


##################################
# Promotions and simplifications
##################################

## PROMOTION RULES

promote_rule(::Type{IdentityOperator{S}}, ::Type{IdentityOperator{T}}) where {S,T} = IdentityOperator{promote_type(S,T)}
promote_rule(::Type{ScalingOperator{S}}, ::Type{IdentityOperator{T}}) where {S,T} = ScalingOperator{promote_type(S,T)}
promote_rule(::Type{DiagonalOperator{S}}, ::Type{IdentityOperator{T}}) where {S,T} = DiagonalOperator{promote_type(S,T)}

promote_rule(::Type{ScalingOperator{S}}, ::Type{ScalingOperator{T}}) where {S,T} = ScalingOperator{promote_type(S,T)}
promote_rule(::Type{DiagonalOperator{S}}, ::Type{ScalingOperator{T}}) where {S,T} = DiagonalOperator{promote_type(S,T)}

promote_rule(::Type{ScalingOperator{S}}, ::Type{ZeroOperator{T}}) where {S,T} = ScalingOperator{promote_type(S,T)}
promote_rule(::Type{DiagonalOperator{S}}, ::Type{ZeroOperator{T}}) where {S,T} = DiagonalOperator{promote_type(S,T)}

## CONVERSIONS

convert(::Type{IdentityOperator{S}}, op::IdentityOperator{T}) where {S,T} = promote_eltype(op, S)
convert(::Type{ScalingOperator{S}}, op::IdentityOperator{T}) where {S,T} =
    ScalingOperator(src(op), dest(op), one(S))
convert(::Type{DiagonalOperator{S}}, op::IdentityOperator{T}) where {S,T} =
    DiagonalOperator(src(op), dest(op), ones(S, length(src(op))))

convert(::Type{ScalingOperator{S}}, op::ScalingOperator{T}) where {S,T} = promote_eltype(op, S)
convert(::Type{DiagonalOperator{S}}, op::ScalingOperator{T}) where {S,T} =
    DiagonalOperator(src(op), dest(op), S(scalar(op))*ones(S,length(src(op))))

convert(::Type{ZeroOperator{S}}, op::ZeroOperator{T}) where {S,T} = promote_eltype(op, S)
convert(::Type{DiagonalOperator{S}}, op::ZeroOperator{T}) where {S,T} =
    DiagonalOperator(src(op), dest(op), zeros(S, length(src(op))))

convert(::Type{DiagonalOperator{S}}, op::DiagonalOperator{T}) where {S,T} = promote_eltype(op, S)


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



(*)(op1::DiagonalOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), diagonal(op1) .* diagonal(op2))
(*)(op1::ScalingOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), scalar(op1) * diagonal(op2))
(*)(op2::DiagonalOperator, op1::ScalingOperator) = op1 * op2

(+)(op1::DiagonalOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), diagonal(op1) + diagonal(op2))
(+)(op1::ScalingOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), scalar(op1) +  diagonal(op2))
(+)(op2::DiagonalOperator, op1::ScalingOperator) = op1 + op2


# """
# A ComplexifyOperator converts real numbers to their complex counterparts.
#
# A ComplexifyOperator applied to a complex basis is simplified to the IdentityOperator.
# """
# struct ComplexifyOperator{T} <: DictionaryOperator{T}
#     src   ::  Dictionary
#     dest  ::  Dictionary
#
#     function ComplexifyOperator{T}(src, dest) where T
#         @assert length(src) == length(dest)
#         @assert complex(eltype(src)) == eltype(dest)
#         new(src, dest)
#     end
# end
#
# ComplexifyOperator(src::Dictionary, dest::Dictionary) = ComplexifyOperator(op_eltype(src, dest), src, dest)
#
# function ComplexifyOperator(::Type{T}, src::Dictionary, dest::Dictionary) = ComplexifyOperator{T}(src, dest)
#     S, D, A = op_eltypes(src, dest, T)
#     ComplexifyOperator{A}(promote_coeftype(src, S), promote_coeftype(dest, D))
# end
#
# similar_operator(::ComplexifyOperator, ::Type{S}, src::Dictionary, dest::Dictionary) = ComplexifyOperator(S, src, dest)
#
# inv(op::ComplexifyOperator) = RealifyOperator(dest(op), src(op))
#
# is_diagonal(::ComplexifyOperator) = true
#
# ctranspose(op::ComplexifyOperator) = inv(op)
#
# function apply!(op::ComplexifyOperator, coef_dest, coef_src)
#     for i in eachindex(coef_src)
#         coef_dest[i] = complex(coef_src[i])
#     end
# end
#

## A RealifyOperator is problematic because it is not a linear operator.
#
# """
# A RealifyOperator converts complex numbers to their real counterparts.
#
# If the complex numbers should have no significant imaginary part.
# A RealifyOperator applied to a real basis is simplified to the IdentityOperator.
# """
# struct RealifyOperator{T} <: DictionaryOperator{T}
#     src   ::  Dictionary
#     dest  ::  Dictionary
#
#     function RealifyOperator{T}(src, dest) where T
#         @assert length(src) == length(dest)
#         @assert real(eltype(src)) == eltype(dest)
#         new(src, dest)
#     end
# end
#
# RealifyOperator(src::Dictionary, dest::Dictionary) = RealifyOperator(op_eltype(src, dest), src, dest)
#
# function RealifyOperator(::Type{T}, src::Dictionary, dest::Dictionary) = RealifyOperator{T}(src, dest)
#     S, D, A = op_eltypes(src, dest, T)
#     RealifyOperator{A}(promote_coeftype(src, S), promote_coeftype(dest, D))
# end
#
# similar_operator(::RealifyOperator, ::Type{S}, src::Dictionary, dest::Dictionary) = RealifyOperator(S, src, dest)
#
#
# inv(op::RealifyOperator) = ComplexifyOperator(dest(op), src(op))
#
# is_diagonal(::RealifyOperator) = true
#
# ctranspose(op::RealifyOperator) = inv(op)
#
# function apply!(op::RealifyOperator, coef_dest, coef_src)
#     exact = true
#     for i in eachindex(coef_src)
#         coef_dest[i] = real(coef_src[i])
#         if !(abs(imag(coef_src[i]))<sqrt(eps(real(eltype(op)))))
#             exact =  false
#         end
#     end
#     !exact && (warn("Realify operator can not realify exactly."))
# end
