
###############
# Arithmetics
###############
# We can not specialize (*/∘/...)(ops::AbstractOperator...), since too general
(*)(op1::AbstractOperator, op2::AbstractOperator) = compose(op2, op1)
(∘)(op1::AbstractOperator, op2::AbstractOperator) = (*)(op1, op2)

# make times (*) a synonym for applying the operator
(*)(op::AbstractOperator, object) = apply(op, object)

mul!(y::AbstractVector, A::DictionaryOperator, x::AbstractVector) = apply!(A, tocoefficientformat(y,dest(A)), tocoefficientformat(x,src(A)))
mul!(y::AbstractMatrix, A::DictionaryOperator, x::AbstractMatrix) = apply_multiple!(A, y, x)



(∘)(op1::ArrayOperator, op2::ArrayOperator) = ArrayOperator(Mul(unsafe_matrix(op1), unsafe_matrix(op2)), src(op2), dest(op1))
(*)(op1::ArrayOperator, op2::ArrayOperator) = ArrayOperator(_checked_mul(unsafe_matrix(op1), unsafe_matrix(op2)), src(op2), dest(op1))
(+)(op1::ArrayOperator, op2::ArrayOperator) = ArrayOperator(_checked_sum(unsafe_matrix(op1), unsafe_matrix(op2)), src(op2), dest(op1))

(*)(a::Number, op::ArrayOperator) = ArrayOperator(a*unsafe_matrix(op), src(op), dest(op))
(*)(op::ArrayOperator, a::Number) = ArrayOperator(a*unsafe_matrix(op), src(op), dest(op))

(*)(a::Number, op::AbstractOperator) = ScalingOperator(dest(op), a) * op
(*)(op::AbstractOperator, a::Number) = op * ScalingOperator(src(op), a)


for op in (:unchecked_mul,:unchecked_sum, :∘)
    # copied from base such that is is sufficient to implement the two argument problem
    @eval ($op)(a, b, c, xs...) = Base.afoldl($op, ($op)(($op)(a,b),c), xs...)
end

# Default implementation, not typestable...
function _checked_mul(a1::AbstractArray, a2::AbstractArray)
    a = a1*a2
    if isefficient(a1) && isefficient(a2) && !isefficient(a) && !(ignore_mul_message(a1, a2, a))
            @warn "The multiplication, $(typeof(a)), of efficient $(typeof(a1)) and efficient $(typeof(a2)) is not efficient.\n
        Consider implementing *(a1::$(typeof(a1)), a2::$(typeof(a2)))"
    end
    a
end

function _checked_sum(a1::AbstractArray, a2::AbstractArray)
    a = a1+a2
    if isefficient(a1) && isefficient(a2) && !isefficient(a) && !(ignore_sum_message(a1, a2, a))
        @warn "The sum, $(typeof(a)), of efficient $(typeof(a1)) and efficient $(typeof(a2)) is not efficient.\n
        Consider implementing +(a1::$(typeof(a1)), a2::$(typeof(a2)))"
    end
    a
end

ignore_mul_message(a, b, c) = false
ignore_sum_message(a, b, c) = false
