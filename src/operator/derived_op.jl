# derived_op.jl

abstract type DerivedOperator{T} <: DictionaryOperator{T}
end

superoperator(op::DerivedOperator) = op.superoperator

for op in (:src, :dest, :isinplace, :isdiag, :diag, :unsafe_diag)
	@eval $op(operator::DerivedOperator) = $op(superoperator(operator))
end

for op in (:matrix!, :apply_inplace!)
	@eval $op(operator::DerivedOperator, a) = $op(superoperator(operator), a)
end

for op in (:apply!,)
	@eval $op(operator::DerivedOperator, coef_dest, coef_src) = $op(superoperator(operator), coef_dest, coef_src)
end

for op in (:unsafe_getindex,)
	@eval $op(operator::DerivedOperator, i, j) = $op(superoperator(operator), i, j)
end

for op in (:inv, :adjoint)
	@eval $op(operator::DerivedOperator) = $op(superoperator(operator))
end


struct ConcreteDerivedOperator{T} <: DerivedOperator{T}
	superoperator		:: DictionaryOperator{T}
end

similar_operator(op::ConcreteDerivedOperator, ::Type{S}, src, dest) where {S} =
	ConcreteDerivedOperator(similar_operator(superoperator(op), S, src, dest))

hasstencil(op::DerivedOperator) = true
stencilarray(op::DerivedOperator) = [modifiersymbol(op), "(", superoperator(op), ")"]
