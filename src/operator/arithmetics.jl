
###############
# Arithmetics
###############

(*)(op1::ArrayOperator, op2::ArrayOperator) = ArrayOperator(op1.A*op2.A, src(op2), dest(op1))
(*)(op1::ArrayOperator, op2::ArrayOperator, op3::ArrayOperator) = ArrayOperator(op1.A*op2.A*op3.A, src(op3), dest(op1))

(+)(op1::ArrayOperator, op2::ArrayOperator) = ArrayOperator(op1.A+op2.A, src(op2), dest(op1))

(*)(a::Number, op::ArrayOperator) = ArrayOperator(a*op.A, src(op), dest(op))
(*)(op::ArrayOperator, a::Number) = ArrayOperator(a*op.A, src(op), dest(op))

(*)(a::Number, op::AbstractOperator) = ScalingOperator(dest(op), a) * op
(*)(op::AbstractOperator, a::Number) = op * ScalingOperator(src(op), a)
