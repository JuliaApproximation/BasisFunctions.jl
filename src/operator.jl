# operator.jl


abstract AbstractOperator{SRC <: AbstractFunctionSet,DEST <: AbstractFunctionSet}

numtype(op::AbstractOperator) = numtype(src(op))
numtype{SRC,DEST}(::Type{AbstractOperator{SRC,DEST}}) = numtype(SRC)
numtype{OP <: AbstractOperator}(::Type{OP}) = numtype(super(OP))

eltype(op::AbstractOperator) = mixed_eltype(eltype(src(op)), eltype(dest(op)))

# Default implementation of src and dest
src(op::AbstractOperator) = op.src

dest(op::AbstractOperator) = op.dest

# The size of the operator as a linear map from source to destination.
# It is equal to the size of its matrix representation.
size(op::AbstractOperator) = (length(dest(op)), length(src(op)))

size(op::AbstractOperator, j::Int) = j==1 ? length(dest(op)) : length(src(op))


function apply(op::AbstractOperator, coef_src)
	coef_dest = Array(eltype(dest(op)), size(dest(op)))
	apply!(op, coef_dest, coef_src)
	coef_dest
end

# This general definition makes it easier to dispatch on source and destination
apply!(op::AbstractOperator, coef_dest, coef_src) = apply!(op, dest(op), src(op), coef_dest, coef_src)

(*)(op::AbstractOperator, coef_src) = apply(op, coef_src)

function matrix(op::AbstractOperator)
	a = Array(eltype(op), size(op))
	matrix!(op, a)
	a
end

function matrix!{T}(op::AbstractOperator, a::Array{T})
	n = length(src(op))
	m = length(dest(op))

	@assert (m,n) == size(a)

	r = zeros(T,n)
	s = zeros(T,m)
	for i = 1:n
		if (i > 1)
			r[i-1] = zero(T)
		end
		r[i] = one(T)
		apply!(op, reshape(s, size(dest(op))), reshape(r, size(src(op))))
		a[:,i] = s
	end
end


# The transpose of an operator
immutable OperatorTranspose{OP <: AbstractOperator,SRC,DEST} <: AbstractOperator{SRC,DEST}
	op	::	OP

	OperatorTranspose(op::AbstractOperator{DEST,SRC}) = new(op)
end

transpose{SRC,DEST}(op::AbstractOperator{DEST,SRC}) = OperatorTranspose{typeof(op),SRC,DEST}(op)

operator(op::OperatorTranspose) = op.op

src(op::OperatorTranspose) = dest(operator(op))

dest(op::OperatorTranspose) = src(operator(op))

apply!(op::OperatorTranspose, coef_dest, coef_src) = apply!(op, operator(op), coef_dest, coef_src)



# A composite operator applies op2 after op1. It preallocates sufficient memory to store intermediate results.
immutable CompositeOperator{OP1 <: AbstractOperator,OP2 <: AbstractOperator,T,N,SRC,DEST} <: AbstractOperator{SRC,DEST}
	op1		::	OP1
	op2		::	OP2
	scratch	::	Array{T,N}	# For storing the intermediate result after applying op1

	CompositeOperator(op1::OP1, op2::OP2) = new(op1, op2, Array(T,size(dest(op1))))
end

CompositeOperator{SRC1,DEST1,SRC2,DEST2}(op1::AbstractOperator{SRC1,DEST1}, op2::AbstractOperator{SRC2,DEST2}) = CompositeOperator{typeof(op1),typeof(op2),eltype(dest(op2)),dim(dest(op2)),SRC1,DEST2}(op1,op2)


src(op::CompositeOperator) = src(op.op1)

dest(op::CompositeOperator) = dest(op.op2)


function apply!(op::CompositeOperator, coef_dest, coef_src, scratch)
	apply!(op.op1, scratch, coef_src)
	apply!(op.op2, coef_dest, scratch)
end

apply!(op::CompositeOperator, coef_dest, coef_src) = apply!(op, coef_dest, coef_src, op.scratch)


# A DenseOperator stores its matrix representation upon construction.
immutable DenseOperator{OP <: AbstractOperator,ELT,SRC,DEST} <: AbstractOperator{SRC,DEST}
	op		::	OP
	matrix	::	Array{ELT,2}

	DenseOperator(op::AbstractOperator{SRC,DEST}) = new(op, matrix(op))
end

DenseOperator{SRC,DEST}(op::AbstractOperator{SRC,DEST}) = DenseOperator{typeof(op),eltype(op),SRC,DEST}(op)

apply!(op::DenseOperator, coef_dest, coef_src) = (coef_dest[:] = op.matrix * coef_src)

matrix(op::DenseOperator) = op.matrix

matrix!(op::AbstractOperator, a::Array) = (a[:] = op.matrix)


# Catch-all for missing implementations
apply!(op::AbstractOperator, dest, src, coef_dest, coef_src) = println("Operation on ", op, " not implemented.")



