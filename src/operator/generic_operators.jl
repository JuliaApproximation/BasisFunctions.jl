# Some generic operators

"The generic identity operator does nothing to the function it is applied to."
struct GenericIdentityOperator <: AbstractOperator
    src_space   ::  FunctionSpace
end

src_space(op::GenericIdentityOperator) = op.src_space
dest_space(op::GenericIdentityOperator) = src_space(op)

apply(op::GenericIdentityOperator, fun) = fun
