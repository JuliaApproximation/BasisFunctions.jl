# differentiation.jl

immutable Differentiation{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
    var     ::  Int
    order   ::  Int
end

Differentiation{SRC,DEST}(src::SRC, dest::DEST, var::Int=1, order::Int=1) = Differentiation{SRC,DEST}(src, dest, var, order)

variable(op::Differentiation) = op.var

order(op::Differentiation) = op.order

# A shortcut routine to compute the derivative of a basis that is closed under differentiation
differentiate(src::AbstractBasis, coef) = apply(Differentiation(src,src), coef)


