# differentiation.jl

immutable DifferentiationOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
    var     ::  Int     
    order   ::  Int
end


variable(op::DifferentiationOperator) = op.var

order(op::DifferentiationOperator) = op.order

differentiate(dest::AbstractBasis, src::AbstractBasis, coef, var) = differentiate(dest, src, coef, var, 1)

differentiate(dest::AbstractBasis1d, src::AbstractBasis1d, coef) = differentiate(dest, src, coef, 1, 1)

function differentiate(dest::AbstractBasis, src::AbstractBasis, coef, var, order)
    result = Array(eltype(dest), length(dest))
    differentiate!(dest, src, result, coef, var, order)
    return result
end


