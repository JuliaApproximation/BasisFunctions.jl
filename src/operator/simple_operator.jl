# simple_operator.jl

## Identity operator

struct UnboundIdentityOperator{T} <: UnboundOperator{T}
end

is_inplace(op::UnboundIdentityOperator) = true
is_diagonal(op::UnboundIdentityOperator) = true

apply_inplace!(op::UnboundIdentityOperator, coefficients) = coefficients

inv(op::UnboundIdentityOperator) = op

ctranspose(op::UnboundIdentityOperator) = op




## Scaling operator

struct UnboundScalingOperator{T} <: UnboundOperator{T}
    scalar  ::  T
end

is_inplace(op::UnboundScalingOperator) = true
is_diagonal(op::UnboundScalingOperator) = true

function apply_inplace!(op::UnboundScalingOperator, coefficients)
    scalar = op.scalar
    for i in eachindex(coefficients)
        coefficients[i] = scalar * coefficients[i]
    end
    coefficients
end

function apply!(op::UnboundScalingOperator, dest, src)
    scalar = op.scalar
    for i in eachindex(src, dest)
        dest[i] = scalar * src[i]
    end
    dest
end

inv(op::UnboundScalingOperator) = UnboundScalingOperator(inv(op.scalar))

ctranspose(op::UnboundScalingOperator) = UnboundScalingOperator(ctranspose(op.scalar))
