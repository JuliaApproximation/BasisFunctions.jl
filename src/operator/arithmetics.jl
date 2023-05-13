
###############
# Arithmetics
###############

# We can not specialize (*/∘/...)(ops::AbstractOperator...), since too general
(*)(op1::AbstractOperator, op2::AbstractOperator) = compose(op2, op1)
(∘)(op1::AbstractOperator, op2::AbstractOperator) = (*)(op1, op2)

(∘)(op1::AbstractMatrix, op2::AbstractOperator) = compose(op2, ArrayOperator(op1, dest(op2)))
(∘)(op1::AbstractOperator, op2::AbstractMatrix) = compose(ArrayOperator(op2, src(op1)), op1)

# make times (*) a synonym for applying the operator
(*)(op::AbstractOperator, object) = apply(op, object)

mul!(y::AbstractVector, A::DictionaryOperator, x::AbstractVector) = apply!(A, tocoefficientformat(y,dest(A)), tocoefficientformat(x,src(A)))
mul!(y::AbstractMatrix, A::DictionaryOperator, x::AbstractMatrix) = apply_multiple!(A, y, x)

# LazyArrays does not work as well as composite operator.
# include("../util/lazyarrays.jl")
# (∘)(op1::ArrayOperator, op2::ArrayOperator) = ArrayOperator(Mul(unsafe_matrix(op1), unsafe_matrix(op2)), src(op2), dest(op1))
# (*)(op1::ArrayOperator, op2::ArrayOperator) = ArrayOperator(_checked_mul(unsafe_matrix(op1), unsafe_matrix(op2)), src(op2), dest(op1))
# (+)(op1::ArrayOperator, op2::ArrayOperator) = ArrayOperator(_checked_sum(unsafe_matrix(op1), unsafe_matrix(op2)), src(op2), dest(op1))
# (-)(op1::ArrayOperator, op2::ArrayOperator) = ArrayOperator(_checked_sum(unsafe_matrix(op1), -unsafe_matrix(op2)), src(op2), dest(op1))

# Hacks for some operators

for (OP) in (:DiagonalOperator, :CirculantOperator, :ScalingOperator)
    @eval begin
        (*)(op1::$OP, op2::$OP) = ArrayOperator(_checked_mul(unsafe_matrix(op1), unsafe_matrix(op2)), src(op2), dest(op1))
        (+)(op1::$OP, op2::$OP) = ArrayOperator(_checked_sum(unsafe_matrix(op1), unsafe_matrix(op2)), src(op2), dest(op1))
        (-)(op1::$OP, op2::$OP) = ArrayOperator(_checked_sum(unsafe_matrix(op1), -unsafe_matrix(op2)), src(op2), dest(op1))
    end
end


(*)(a::Number, op::ArrayOperator) = ArrayOperator(a*unsafe_matrix(op), src(op), dest(op))
(*)(op::ArrayOperator, a::Number) = ArrayOperator(a*unsafe_matrix(op), src(op), dest(op))

(*)(a::Number, op::DictionaryOperator) = ScalingOperator(dest(op), a) * op
(*)(op::DictionaryOperator, a::Number) = op * ScalingOperator(src(op), a)


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

## Linear algebra routines

LinearAlgebra.eigvals(op::DictionaryOperator; options...) = eigvals(matrix(op); options...)
LinearAlgebra.svd(op::DictionaryOperator; options...) = svd(matrix(op); options...)
LinearAlgebra.qr(op::DictionaryOperator; options...) = qr(matrix(op); options...)
LinearAlgebra.eigen(op::DictionaryOperator; options...) = eigen(matrix(op); options...)
LinearAlgebra.norm(op::DictionaryOperator; options...) = norm(matrix(op); options...)
LinearAlgebra.rank(op::DictionaryOperator; options...) = rank(matrix(op); options...)
# specialize \ only when the operator has an underlying array type
Base.:\(op::ArrayOperator, rhs::AbstractVector) = matrix(op) \ rhs
