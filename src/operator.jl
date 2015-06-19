# operator.jl


abstract AbstractOperator{SRC,DEST}

numtype(op::AbstractOperator) = numtype(src(op))
numtype{SRC,DEST}(::Type{AbstractOperator{SRC,DEST}}) = numtype(SRC)
numtype{OP <: AbstractOperator}(::Type{OP}) = numtype(super(OP))

eltype(op::AbstractOperator) = promote_type(eltype(src(op)), eltype(dest(op)))

eltype(op1::AbstractOperator, op::AbstractOperator...) = promote_type(eltype(op1), map(eltype, op)...)

eltype(b1::AbstractFunctionSet, b::AbstractFunctionSet...) = promote_type(eltype(b1), map(eltype, b)...)


# Default implementation of src and dest
src(op::AbstractOperator) = op.src
dest(op::AbstractOperator) = op.dest

# The size of the operator as a linear map from source to destination.
# It is equal to the size of its matrix representation.
size(op::AbstractOperator) = (length(dest(op)), length(src(op)))

size(op::AbstractOperator, j::Int) = j==1 ? length(dest(op)) : length(src(op))

is_inplace(op::AbstractOperator) = False()


function apply(op::AbstractOperator, coef_src)
	coef_dest = Array(eltype(op), size(dest(op)))
	apply!(op, coef_dest, coef_src)
	coef_dest
end

# This general definition makes it easier to dispatch on source and destination.
# Operators can choose to specialize with or without the src and dest arguments.
apply!(op::AbstractOperator, coef_dest, coef_src) = _apply!(op, is_inplace(op), coef_dest, coef_src)

# Operator is in-place, use its in-place operation but don't overwrite coef_src
function _apply!(op::AbstractOperator, op_inplace::True, coef_dest, coef_src)
	for i in eachindex(coef_src)
		coef_dest[i] = coef_src[i]
	end
	apply!(op, coef_dest)
end

# This general definition makes it easier to dispatch on source and destination.
# Operators can choose to specialize with or without the src and dest arguments.
_apply!(op::AbstractOperator, op_inplace::False, coef_dest, coef_src) = apply!(op, dest(op), src(op), coef_dest, coef_src)

# Provide a general dispatchable definition for in-place operators also
apply!(op::AbstractOperator, coef_srcdest) = apply!(op, dest(op), src(op), coef_srcdest)

## # Catch-all for missing implementations
## apply!(op::AbstractOperator, dest, src, coef_dest, coef_src) = println("Operation of ", op, " not implemented.")

# Catch-all for missing implementations
apply!(op::AbstractOperator, dest, src, coef_srcdest) = println("In-place operation of ", op, " not implemented.")

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

ctranspose{SRC,DEST}(op::AbstractOperator{DEST,SRC}) = OperatorTranspose{typeof(op),SRC,DEST}(op)

operator(opt::OperatorTranspose) = opt.op

src(opt::OperatorTranspose) = dest(operator(opt))

dest(opt::OperatorTranspose) = src(operator(opt))

is_inplace(opt::OperatorTranspose) = is_inplace(operator(opt))

apply!(opt::OperatorTranspose, coef_dest, coef_src) = apply!(opt, operator(opt), coef_dest, coef_src)

# Definition to make dispatch on source and destination possible.
apply!(opt::OperatorTranspose, op::AbstractOperator, coef_dest, coef_src) = apply!(opt, op, dest(opt), src(opt), coef_dest, coef_src)



# The identity operator
immutable IdentityOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

IdentityOperator(src) = IdentityOperator(src, src)

is_inplace(op::IdentityOperator) = True()

apply!(op::IdentityOperator, dest, src, coef_srcdest) = nothing


# The identity operator up to a scaling
immutable ScalingOperator{T,SRC,DEST} <: AbstractOperator{SRC,DEST}
	scalar	::	T
	src		::	SRC
	dest	::	DEST
end

is_inplace(op::ScalingOperator) = True()

scalar(op::ScalingOperator) = op.scalar

(*){T <: Number}(a::T, op::IdentityOperator) = ScalingOperator(a, src(op), dest(op))
(*){T <: Number}(op::IdentityOperator, a::T) = ScalingOperator(a, src(op), dest(op))

function apply!(op::ScalingOperator, dest, src, coef_srcdest)
	for i in eachindex(coef_srcdest)
		coef_srcdest[i] *= op.scalar
        end
end

# Extra definition to avoid making two passes over the data
function apply!(op::ScalingOperator, dest, src, coef_dest, coef_src)
	for i in eachindex(coef_src)
		coef_dest[i] = op.scalar * coef_src[i]
	end
end


# A composite operator applies op2 after op1. It preallocates sufficient memory to store intermediate results.
immutable CompositeOperator{OP1 <: AbstractOperator,OP2 <: AbstractOperator,T,N,SRC,DEST} <: AbstractOperator{SRC,DEST}
	op1		::	OP1
	op2		::	OP2
	scratch	::	Array{T,N}	# For storing the intermediate result after applying op1

	function CompositeOperator(op1::OP1, op2::OP2)
		@assert size(op1,1) == size(op2,2)

		new(op1, op2, zeros(T,size(src(op2))))
	end
end


# We could ask that DEST1 == SRC2 but that might be too strict. As long as the operators are compatible things are fine.
CompositeOperator{SRC1,DEST1,SRC2,DEST2}(op1::AbstractOperator{SRC1,DEST1}, op2::AbstractOperator{SRC2,DEST2}) = CompositeOperator{typeof(op1),typeof(op2),eltype(dest(op1)),length(size(src(op2))),SRC1,DEST2}(op1,op2)

src(op::CompositeOperator) = src(op.op1)

dest(op::CompositeOperator) = dest(op.op2)

eltype(op::CompositeOperator) = eltype(op.op1, op.op2)

(*)(op2::AbstractOperator, op1::AbstractOperator) = CompositeOperator(op1, op2)

apply!(op::CompositeOperator, coef_dest, coef_src) = _apply!(op, is_inplace(op.op2), coef_dest, coef_src)


function _apply!(op::CompositeOperator, op2_inplace::True, coef_dest, coef_src)
	apply!(op.op1, coef_dest, coef_src)
	apply!(op.op2, coef_dest)
end

function _apply!(op::CompositeOperator, op2_inplace::False, coef_dest, coef_src)
	apply!(op.op1, op.scratch, coef_src)
	apply!(op.op2, coef_dest, op.scratch)
end


# If it is called in-place, assuming all operators support in-place operations.
function apply!(op::CompositeOperator, coef_srcdest)
	apply!(op.op1, coef_srcdest)
	apply!(op.op2, coef_srcdest)
end

# Perhaps the parameters are excessive in this and many other types in the code. Do some tests without them later.
immutable TripleCompositeOperator{OP1 <: AbstractOperator,OP2 <: AbstractOperator,OP3 <: AbstractOperator,T,N1,N2,SRC,DEST} <: AbstractOperator{SRC,DEST}
	op1			::	OP1
	op2			::	OP2
	op3			::	OP3
	scratch1	::	Array{T,N1}	# For storing the intermediate result after applying op1
	scratch2	::	Array{T,N2}	# For storing the intermediate result after applying op2

	function TripleCompositeOperator(op1::OP1, op2::OP2, op3::OP3)
		@assert size(op1,1) == size(op2,2)
		@assert size(op2,1) == size(op3,2)

		new(op1, op2, op3, zeros(T,size(src(op2))), zeros(T,size(src(op3))))
	end
end

TripleCompositeOperator{SRC1,DEST1,SRC3,DEST3}(op1::AbstractOperator{SRC1,DEST1}, op2, op3::AbstractOperator{SRC3,DEST3}) =
	TripleCompositeOperator{typeof(op1),typeof(op2),typeof(op3),eltype(op1,op2,op3),length(size(src(op2))),length(size(src(op3))),SRC1,DEST3}(op1, op2, op3)

src(op::TripleCompositeOperator) = src(op.op1)

dest(op::TripleCompositeOperator) = dest(op.op3)

eltype(op::TripleCompositeOperator) = eltype(op.op1, op.op2, op.op3)

apply!(op::TripleCompositeOperator, coef_dest, coef_src) = _apply!(op, is_inplace(op.op2), is_inplace(op.op3), coef_dest, coef_src)

function _apply!(op::TripleCompositeOperator, op2_inplace::True, op3_inplace::True, coef_dest, coef_src)
	apply!(op.op, coef_dest, coef_src)
	apply!(op.op2, coef_dest)
	apply!(op.op3, coef_dest)
end

function _apply!(op::TripleCompositeOperator, op2_inplace::True, op3_inplace::False, coef_dest, coef_src)
	apply!(op.op1, op.scratch2, coef_src)
	apply!(op.op2, op.scratch2)
	apply!(op.op3, coef_dest, op.scratch2)
end

function _apply!(op::TripleCompositeOperator, op2_inplace::False, op3_inplace::True, coef_dest, coef_src)
	apply!(op.op1, op.scratch1, coef_src)
	apply!(op.op2, coef_dest, op.scratch1)
	apply!(op.op3, coef_dest)
end

function _apply!(op::TripleCompositeOperator, op2_inplace::False, op3_inplace::False, coef_dest, coef_src)
	apply!(op.op1, op.scratch1, coef_src)
	apply!(op.op2, op.scratch2, op.scratch1)
	apply!(op.op3, coef_dest, op.scratch2)
end


# If it is called in-place, assuming all operators support in-place operations.
function apply!(op::TripleCompositeOperator, coef_srcdest)
	apply!(op.op1, coef_srcdest)
	apply!(op.op2, coef_srcdest)
	apply!(op.op3, coef_srcdest)
end


(*)(op3::AbstractOperator, op2::AbstractOperator, op1::AbstractOperator) = TripleCompositeOperator(op1, op2, op3)



# A linear combination of operators: val1 * op1 + val2 * op2.
immutable OperatorSum{OP1 <: AbstractOperator,OP2 <: AbstractOperator,T,N,SRC,DEST} <: AbstractOperator{SRC,DEST}
	op1			::	OP1
	op2			::	OP2
	val1		::	T
	val2		::	T
	scratch		::	Array{T,N}

	function OperatorSum(op1::OP1, op2::OP2, val1::T, val2::T)
		@assert size(op1) == size(op2)
		new(op1, op2, val1, val2, zeros(T,size(dest(op1))))
	end
end

OperatorSum{SRC,DEST}(op1::AbstractOperator{SRC,DEST}, op2::AbstractOperator, val1, val2) =
	OperatorSum{typeof(op1), typeof(op2), eltype(op1,op2), length(size(dest(op1))), SRC, DEST}(op1, op2, promote(val1, val2)...)

src(op::OperatorSum) = src(op.op1)

dest(op::OperatorSum) = dest(op.op1)

eltype{OP1,OP2,T}(op::OperatorSum{OP1,OP2,T}) = T


apply!(op::OperatorSum, coef_srcdest) = apply!(op, op.op1, op.op2, coef_srcdest)

function apply!(op::OperatorSum, op1, op2, coef_srcdest)
	scratch = op.scratch

	apply!(op1, scratch, coef_srcdest)
	apply!(op2, coef_srcdest)

	for i in eachindex(coef_srcdest)
		coef_srcdest[i] = op.val1 * scratch[i] + op.val2 * coef_srcdest[i]
	end
end

apply!(op::OperatorSum, coef_dest, coef_src) = apply!(op, op.op1, op.op2, coef_dest, coef_src)

function apply!(op::OperatorSum, op1, op2, coef_dest, coef_src)
	scratch = op.scratch

	apply!(op1, scratch, coef_src)
	apply!(op2, coef_dest, coef_src)

	for i in eachindex(coef_dest)
		coef_dest[i] = op.val1 * scratch[i] + op.val2 * coef_dest[i]
	end
end

function apply!(op::OperatorSum, op1::ScalingOperator, op2::ScalingOperator, coef_dest, coef_src)
	val = op.val1 * scalar(op1) + op.val2 * scalar(op2)
	for i in eachindex(coef_dest)
		coef_dest[i] = val * coef_src[i]
	end
end

function apply!(op::OperatorSum, op1::ScalingOperator, op2, coef_dest, coef_src)
	apply!(op2, coef_dest, coef_src)

	val1 = op.val1 * scalar(op1)
	for i in eachindex(coef_dest)
		coef_dest[i] = val1 * coef_src[i] + op.val2 * coef_dest[i]
	end
end

function apply!(op::OperatorSum, op1, op2::ScalingOperator, coef_dest, coef_src)
	apply!(op1, coef_dest, coef_src)

	val2 = op.val2 * scalar(op2)
	for i in eachindex(coef_dest)
		coef_dest[i] = op.val1 * coef_dest[i] + val2 * coef_src[i]
	end
end


(-)(op1::AbstractOperator, op2::AbstractOperator) = OperatorSum(op1, op2, one(eltype(op1)), -one(eltype(op2)))


immutable AffineMap{T,SRC,DEST} <: AbstractOperator{SRC,DEST}
	src		::	SRC
	dest	::	DEST
	matrix	::	Array{T,2}
end

AffineMap{T <: Number}(matrix::Array{T,2}) = AffineMap(Rn{T}(size(matrix,2)), Rn{T}(size(matrix,1)), matrix)

AffineMap{T <: Number}(matrix::Array{Complex{T},2}) = AffineMap(Cn{T}(size(matrix,2)), Cn{T}(size(matrix,1)), matrix)

apply!(op::AffineMap, coef_dest, coef_src) = (coef_dest[:] = op.matrix * coef_src)


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





