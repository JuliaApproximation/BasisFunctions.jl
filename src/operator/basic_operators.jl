# basic_operators.jl

"""
The identity operator between two (possibly different) function sets.
"""
immutable IdentityOperator{T} <: AbstractOperator{T}
    src     ::  FunctionSet
    dest    ::  FunctionSet

    function IdentityOperator(src, dest)
        @assert length(src) == length(dest)
        new(src, dest)
    end
end

IdentityOperator{N1,N2,T}(src::FunctionSet{N1,T}, dest::FunctionSet{N2,T}) =
    IdentityOperator{T}(src, dest)

IdentityOperator(src, dest = src) = IdentityOperator(promote_eltype(src,dest)...)

op_promote_eltype{T,S}(op::IdentityOperator{T}, ::Type{S}) =
    IdentityOperator{S}(promote_eltypes(S, src(op), dest(op))...)

@add_properties(IdentityOperator, is_inplace, is_diagonal)

inv(op::IdentityOperator) = IdentityOperator(dest(op), src(op))

ctranspose(op::IdentityOperator) = IdentityOperator(dest(op), src(op))

function matrix!(op::IdentityOperator, a)
    @assert size(a,1) == size(a,2)

    a[:] = 0
    for i in 1:size(a,1)
        a[i,i] = 1
    end
    a
end

diagonal(op::IdentityOperator) = ones(eltype(op), length(src(op)))

unsafe_diagonal(op::IdentityOperator, i) = one(eltype(op))

apply_inplace!(op::IdentityOperator, coef_srcdest) = coef_srcdest


"""
A ScalingOperator is the identity operator up to a scaling.
"""
immutable ScalingOperator{T} <: AbstractOperator{T}
    src     ::  FunctionSet
    dest    ::  FunctionSet
    scalar  ::  T

    function ScalingOperator(src, dest, scalar)
        @assert length(src) == length(dest)
        new(src, dest, scalar)
    end
end

function ScalingOperator(src::FunctionSet, dest::FunctionSet, scalar::Number)
    T = promote_type(eltype(src), eltype(dest), typeof(scalar))
    ScalingOperator{T}(promote_eltypes(T, src, dest)..., convert(T, scalar))
end

ScalingOperator(src::FunctionSet, scalar::Number) = ScalingOperator(src, src, scalar)

op_promote_eltype{T,S}(op::ScalingOperator{T}, ::Type{S}) =
    ScalingOperator{S}(promote_eltypes(S, src(op), dest(op))..., convert(S, op.scalar))

is_inplace(::ScalingOperator) = true
is_diagonal(::ScalingOperator) = true

scalar(op::ScalingOperator) = op.scalar

ctranspose(op::ScalingOperator) = ScalingOperator(dest(op), src(op), conj(scalar(op)))

inv(op::ScalingOperator) = ScalingOperator(dest(op), src(op), 1/scalar(op))

function apply_inplace!(op::ScalingOperator, coef_srcdest)
    for i in eachindex(coef_srcdest)
        coef_srcdest[i] *= op.scalar
    end
    coef_srcdest
end

# Extra definition for out-of-place version to avoid making an intermediate copy
function apply!(op::ScalingOperator, coef_dest, coef_src)
    for i in eachindex(coef_src, coef_dest)
        # Can we assume that coef_dest and coef_src can both be indexed by i?
        # TODO: do we need eachindex(coef_src, coef_dest)?
        coef_dest[i] = op.scalar * coef_src[i]
    end
    coef_dest
end

for op in (:+, :*, :-, :/)
    @eval $op(op1::ScalingOperator, op2::ScalingOperator) =
        (@assert size(op1) == size(op2); ScalingOperator(src(op1), dest(op1), $op(scalar(op1),scalar(op2))))
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

unsafe_getindex{T}(op::ScalingOperator{T}, i, j) = i == j ? convert(T, op.scalar) : convert(T, 0)


# default implementation for scalar multiplication is a scaling operator
*(scalar::Number, op::AbstractOperator) = ScalingOperator(dest(op),scalar) * op


"The zero operator maps everything to zero."
immutable ZeroOperator{T} <: AbstractOperator{T}
    src     ::  FunctionSet
    dest    ::  FunctionSet
end

ZeroOperator{N1,N2,T}(src::FunctionSet{N1,T}, dest::FunctionSet{N2,T}) =
    ZeroOperator{T}(src, dest)

ZeroOperator(src, dest = src) = ZeroOperator(promote_eltype(src, dest)...)

op_promote_eltype{T,S}(op::ZeroOperator{T}, ::Type{S}) =
    ZeroOperator(promote_eltypes(S, op.src, op.dest)...)

# We can only be in-place if the numbers of coefficients of src and dest match
is_inplace(op::ZeroOperator) = length(src(op))==length(dest(op))

is_diagonal(::ZeroOperator) = true

ctranspose(op::ZeroOperator) = ZeroOperator(dest(op), src(op))

matrix!(op::ZeroOperator, a) = (fill!(a, 0); a)

apply_inplace!(op::ZeroOperator, coef_srcdest) = (fill!(coef_srcdest, 0); coef_srcdest)

apply!(op::ZeroOperator, coef_dest, coef_src) = (fill!(coef_dest, 0); coef_dest)

diagonal(op::ZeroOperator) = zeros(eltype(op), min(length(src(op)), length(dest(op))))

unsafe_diagonal(op::ZeroOperator, i) = zero(eltype(op))

unsafe_getindex{T}(op::ZeroOperator{T}, i, j) = convert(T, 0)





"""
A diagonal operator is represented by a diagonal matrix.

Several other operators can be converted into a diagonal matrix, and this
conversion happens automatically when such operators are combined into a composite
operator.
"""
immutable DiagonalOperator{ELT} <: AbstractOperator{ELT}
    src         ::  FunctionSet
    dest        ::  FunctionSet
    # We store the diagonal in a vector
    diagonal    ::  Vector{ELT}
end

DiagonalOperator{T <: Real}(diagonal::AbstractVector{T}) = DiagonalOperator(Rn{T}(length(diagonal)), diagonal)
DiagonalOperator{T <: Complex}(diagonal::AbstractVector{T}) = DiagonalOperator(Cn{T}(length(diagonal)), diagonal)

DiagonalOperator{ELT}(src::FunctionSet, diagonal::AbstractVector{ELT}) = DiagonalOperator{ELT}(src, src, diagonal)

op_promote_eltype{ELT,S}(op::DiagonalOperator{ELT}, ::Type{S}) =
    DiagonalOperator{S}(promote_eltypes(S, src(op), dest(op))..., convert(Array{S,1}, op.diagonal))

diagonal(op::DiagonalOperator) = copy(op.diagonal)

is_inplace(::DiagonalOperator) = true
is_diagonal(::DiagonalOperator) = true

inv(op::DiagonalOperator) = DiagonalOperator(dest(op), src(op), op.diagonal.^(-1))

ctranspose(op::DiagonalOperator) = DiagonalOperator(dest(op), src(op), conj(diagonal(op)))

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

promote_rule{S,T}(::Type{IdentityOperator{S}}, ::Type{IdentityOperator{T}}) = IdentityOperator{promote_type(S,T)}
promote_rule{S,T}(::Type{ScalingOperator{S}}, ::Type{IdentityOperator{T}}) = ScalingOperator{promote_type(S,T)}
promote_rule{S,T}(::Type{DiagonalOperator{S}}, ::Type{IdentityOperator{T}}) = DiagonalOperator{promote_type(S,T)}

promote_rule{S,T}(::Type{ScalingOperator{S}}, ::Type{ScalingOperator{T}}) = ScalingOperator{promote_type(S,T)}
promote_rule{S,T}(::Type{DiagonalOperator{S}}, ::Type{ScalingOperator{T}}) = DiagonalOperator{promote_type(S,T)}

promote_rule{S,T}(::Type{ScalingOperator{S}}, ::Type{ZeroOperator{T}}) = ScalingOperator{promote_type(S,T)}
promote_rule{S,T}(::Type{DiagonalOperator{S}}, ::Type{ZeroOperator{T}}) = DiagonalOperator{promote_type(S,T)}

## CONVERSIONS

convert{S,T}(::Type{IdentityOperator{S}}, op::IdentityOperator{T}) = promote_eltype(op, S)
convert{S,T}(::Type{ScalingOperator{S}}, op::IdentityOperator{T}) =
    ScalingOperator(src(op), dest(op), one(S))
convert{S,T}(::Type{DiagonalOperator{S}}, op::IdentityOperator{T}) =
    DiagonalOperator(src(op), dest(op), ones(S,length(src(op))))

convert{S,T}(::Type{ScalingOperator{S}}, op::ScalingOperator{T}) = promote_eltype(op, S)
convert{S,T}(::Type{DiagonalOperator{S}}, op::ScalingOperator{T}) =
    DiagonalOperator(src(op), dest(op), S(scalar(op))*ones(S,length(src(op))))

convert{S,T}(::Type{ZeroOperator{S}}, op::ZeroOperator{T}) = promote_eltype(op, S)
convert{S,T}(::Type{DiagonalOperator{S}}, op::ZeroOperator{T}) =
    DiagonalOperator(src(op), dest(op), zeros(S,length(src(op))))

convert{S,T}(::Type{DiagonalOperator{S}}, op::DiagonalOperator{T}) = promote_eltype(op, S)


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
simplify(op1::IdentityOperator, op2::AbstractOperator) = (op2,)
simplify(op1::AbstractOperator, op2::IdentityOperator) = (op1,)

(*)(a::Number, op::IdentityOperator) = ScalingOperator(src(op), dest(op), a)
(*)(op::IdentityOperator, a::Number) = a*op

# The zero operator annihilates all other operators
simplify(op1::ZeroOperator, op2::ZeroOperator) = (op1,)
simplify(op1::ZeroOperator, op2::AbstractOperator) = (op1,)
simplify(op1::AbstractOperator, op2::ZeroOperator) = (op2,)



(*)(op1::DiagonalOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), diagonal(op1) .* diagonal(op2))
(*)(op1::ScalingOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), scalar(op1) * diagonal(op2))
(*)(op2::DiagonalOperator, op1::ScalingOperator) = op1 * op2

(+)(op1::DiagonalOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), diagonal(op1) + diagonal(op2))
(+)(op1::ScalingOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), scalar(op1) +  diagonal(op2))
(+)(op2::DiagonalOperator, op1::ScalingOperator) = op1 + op2


"""
A ComplexifyOperator converts real numbers to their complex counterparts.

A ComplexifyOperator applied to a complex basis is simplified to the IdentityOperator.
"""
immutable ComplexifyOperator{T} <: AbstractOperator{T}
  src   ::  FunctionSet
  dest  ::  FunctionSet
  function ComplexifyOperator(src, dest)
    @assert length(src) == length(dest)
    @assert complex(eltype(src)) == eltype(dest)
    new(src, dest)
  end
end
ComplexifyOperator(src::FunctionSet, dest::FunctionSet) = ComplexifyOperator{eltype(src)}(src, dest)
ComplexifyOperator(src::FunctionSet) = ComplexifyOperator(src, promote_eltype(src,complex(eltype(src))))
ComplexifyOperator{B<:FunctionSet}(src::B, dest::B) = IdentityOperator(src, dest)
Base.inv(op::ComplexifyOperator) = RealifyOperator(dest(op),src(op))
is_diagonal(::ComplexifyOperator) = true
ctranspose(op::ComplexifyOperator) = inv(op)

function apply!(op::ComplexifyOperator, coef_dest, coef_src)
  for i in eachindex(coef_src)
      coef_dest[i] = complex(coef_src[i])
  end
end

"""
A RealifyOperator converts complex numbers to their real counterparts.

If the complex numbers should have no significant imaginary part.
A RealifyOperator applied to a real basis is simplified to the IdentityOperator.
"""
immutable RealifyOperator{T} <: AbstractOperator{T}
  src   ::  FunctionSet
  dest  ::  FunctionSet
  function RealifyOperator(src, dest)
    @assert length(src) == length(dest)
    @assert real(eltype(src)) == eltype(dest)
    new(src, dest)
  end
end
RealifyOperator(src::FunctionSet, dest::FunctionSet) = RealifyOperator{eltype(dest)}(src, dest)
RealifyOperator(src::FunctionSet) = RealifyOperator(src, set_promote_eltype(src,real(eltype(src))))
RealifyOperator{B<:FunctionSet}(src::B, dest::B) = IdentityOperator(src, dest)
inv(op::RealifyOperator) = ComplexifyOperator(dest(op),src(op))
is_diagonal(::RealifyOperator) = true
ctranspose(op::RealifyOperator) = inv(op)
function apply!(op::RealifyOperator, coef_dest, coef_src)
  exact = true
  for i in eachindex(coef_src)
      coef_dest[i] = real(coef_src[i])
      if !(abs(imag(coef_src[i]))<sqrt(eps(real(eltype(op)))))
        exact =  false
      end
  end
  !exact && (warn("Realify operator can not realify exactly."))
end
