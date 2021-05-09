# Some generic operators

"The generic identity operator does nothing to the function it is applied to."
struct GenericIdentityOperator <: AbstractOperator
    src_space   ::  FunctionSpace
end

src_space(op::GenericIdentityOperator) = op.src_space
dest_space(op::GenericIdentityOperator) = src_space(op)

apply(op::GenericIdentityOperator, fun; opts...) = fun


struct GenericScalingOperator <: AbstractOperator
    src_space   ::  FunctionSpace
    scalar      ::  Number
end

src_space(op::GenericScalingOperator) = op.src_space
dest_space(op::GenericScalingOperator) = src_space(op)

scalar(op::GenericScalingOperator) = op.scalar

apply(op::GenericScalingOperator, fun; opts...) = apply(scalar(op),fun)
apply(a::Number, f::Function) = (x->a*f(x))
apply(a::Number, f) = a*f

(*)(a::Number, op::AbstractOperator) = GenericScalingOperator(dest_space(op), a)*op
(*)(op::AbstractOperator, a::Number) = GenericScalingOperator(dest_space(op), a)*op
(*)(op::GenericScalingOperator, I::GenericIdentityOperator) = op
(*)(I::GenericIdentityOperator, op::GenericScalingOperator) = op
(-)(op::AbstractOperator) = (-1)*op


struct GenericSumOperator <: AbstractOperator
    op1   ::    AbstractOperator
    op2   ::    AbstractOperator
end

src_space(op::GenericSumOperator) = src_space(op.op1)
dest_space(op::GenericSumOperator) = dest_space(op.op1)

components(op::GenericSumOperator) = (op1,op2)

apply(op::GenericSumOperator, fun; opts...) = apply(op.op1,fun; opts...) + apply(op.op2,fun; opts...)


scalar(U::UniformScaling) = scalar(U,U.λ)
scalar(::UniformScaling, ::Bool) = 1
scalar(::UniformScaling, a) = a

(+)(op1::AbstractOperator, op2::AbstractOperator) = GenericSumOperator(op1,op2)
(-)(op1::AbstractOperator, op2::AbstractOperator) = GenericSumOperator(op1,-op2)
(+)(I1::UniformScaling, op2::AbstractOperator) = (+)(GenericScalingOperator(dest_space(op2),scalar(I1)), op2)
(-)(I1::UniformScaling, op2::AbstractOperator) = (-)(GenericScalingOperator(dest_space(op2),scalar(I1)), op2)
(+)(op1::BasisFunctions.AbstractOperator, I2::UniformScaling) = (+)(op1, GenericScalingOperator(dest_space(op1),scalar(I2)))
(-)(op1::BasisFunctions.AbstractOperator, I2::UniformScaling) = (-)(op1, GenericScalingOperator(dest_space(op1),scalar(I2)))
