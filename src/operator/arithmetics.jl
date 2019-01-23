
###############
# Arithmetics
###############

(∘)(ops::ArrayOperator...) = ArrayOperator(Mul(map(unsafe_matrix, ops)...), src(ops[end]), dest(ops[1]))
(*)(ops::ArrayOperator...) = ArrayOperator(*(map(unsafe_matrix, ops)...), src(ops[end]), dest(ops[1]))
(+)(ops::ArrayOperator...) = ArrayOperator(+(map(unsafe_matrix, ops)...), src(ops[end]), dest(ops[1]))

(*)(a::Number, op::ArrayOperator) = ArrayOperator(a*unsafe_matrix(op), src(op), dest(op))
(*)(op::ArrayOperator, a::Number) = ArrayOperator(a*unsafe_matrix(op), src(op), dest(op))

(*)(a::Number, op::AbstractOperator) = ScalingOperator(dest(op), a) * op
(*)(op::AbstractOperator, a::Number) = op * ScalingOperator(src(op), a)

LazyArrays.checkdimensions(A::UniformScaling, B, C...) =
    LazyArrays.checkdimensions(B, C...)

LazyArrays.checkdimensions(A, B::UniformScaling, C...) =
    LazyArrays.checkdimensions(A, C...)

LazyArrays.checkdimensions(A::UniformScaling, B::UniformScaling, C...) =
    LazyArrays.checkdimensions(C...)

# LazyArrays.MemoryLayout(::LinearAlgebra.UniformScaling) = LazyArrays.ScalarLayout()
