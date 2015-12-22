# operator.jl


"""
AbstractOperator represents any linear operator that maps SRC to DEST.
Typically, SRC and DEST are of type FunctionSet, but that is not enforced.
The action of the operator is defined by providing a method for apply!.

The dimension of an operator are like a matrix: (length(dest),length(src)).
"""
abstract AbstractOperator{SRC,DEST}

# We inherit the numtype from the source. Numtypes of source and destination should always match.
numtype{SRC,DEST}(::Type{AbstractOperator{SRC,DEST}}) = numtype(SRC)
numtype{OP <: AbstractOperator}(::Type{OP}) = numtype(super(OP))

eltype{SRC,DEST}(::Type{AbstractOperator{SRC,DEST}}) = promote_type(eltype(SRC),eltype(DEST))
eltype{OP <: AbstractOperator}(::Type{OP}) = eltype(super(OP))


# Default implementation of src and dest
src(op::AbstractOperator) = op.src
dest(op::AbstractOperator) = op.dest

# The size of the operator as a linear map from source to destination.
# It is equal to the size of its matrix representation.
size(op::AbstractOperator) = (length(dest(op)), length(src(op)))

size(op::AbstractOperator, j::Int) = j==1 ? length(dest(op)) : length(src(op))

+(op1::AbstractOperator, op2::AbstractOperator) = +(promote(op1,op2)...)

"Trait to indicate whether or not an operator performs its action in place."
is_inplace{OP <: AbstractOperator}(::Type{OP}) = False
is_inplace(op::AbstractOperator) = is_inplace(typeof(op))()

"Trait to indicate whether or not an operator is diagonal."
is_diagonal{OP <: AbstractOperator}(::Type{OP}) = False
is_diagonal(op::AbstractOperator) = is_diagonal(typeof(op))()


function apply(op::AbstractOperator, coef_src)
	coef_dest = Array(promote_type(eltype(op),eltype(coef_src)), size(dest(op)))
	apply!(op, coef_dest, coef_src)
	coef_dest
end

# The function apply(operator,...) by default calls apply(operator, dest, src, ...)
# This general definition makes it easier to dispatch on source and destination.
# Operators can choose to specialize with or without the src and dest arguments.
# In-place operators can be called with a single set of coefficients.
function apply!(op::AbstractOperator, coef_dest, coef_src)
	@assert length(coef_dest) == length(dest(op))
	@assert length(coef_src) == length(src(op))

	# distinguish between operators that are in-place and operators that are not
	_apply!(op, is_inplace(op), coef_dest, coef_src)
end

# Operator is in-place, use its in-place operation but don't overwrite coef_src
function _apply!(op::AbstractOperator, op_inplace::True, coef_dest, coef_src)
	for i in eachindex(coef_src)
		coef_dest[i] = coef_src[i]
	end
	apply!(op, coef_dest)
end

_apply!(op::AbstractOperator, op_inplace::False, coef_dest, coef_src) = apply!(op, dest(op), src(op), coef_dest, coef_src)

# Provide a general dispatchable definition for in-place operators also
apply!(op::AbstractOperator, coef_srcdest) = apply!(op, dest(op), src(op), coef_srcdest)

# Catch-all for missing implementations
apply!(op::AbstractOperator, dest, src, coef_dest, coef_src) = println("Operation of ", op, " not implemented.")

# Catch-all for missing implementations
apply!(op::AbstractOperator, dest, src, coef_srcdest) = println("In-place operation of ", op, " not implemented.")

(*)(op::AbstractOperator, coef_src::AbstractArray) = apply(op, coef_src)


collect(op::AbstractOperator) = matrix(op)

function matrix(op::AbstractOperator)
    a = Array(eltype(op), size(op))
    matrix!(op, a)
    a
end

function matrix!{T}(op::AbstractOperator, a::AbstractArray{T})
    n = length(src(op))
    m = length(dest(op))
    
    @assert (m,n) == size(a)
    
    r = zeros(T,n)
    r_src = reshape(r, size(src(op)))
    s = zeros(T,m)
    s_dest = reshape(s, size(dest(op)))
    for i = 1:n
        if (i > 1)
            r[i-1] = zero(T)
        end
        r[i] = one(T)
        apply!(op, s_dest, r_src)
        a[:,i] = s
    end
end


"An OperatorTranspose represents the transpose of an operator."
immutable OperatorTranspose{OP,SRC,DEST} <: AbstractOperator{SRC,DEST}
	op	::	OP

	OperatorTranspose(op::AbstractOperator{DEST,SRC}) = new(op)
end

ctranspose{SRC,DEST}(op::AbstractOperator{DEST,SRC}) = OperatorTranspose{typeof(op),SRC,DEST}(op)

operator(opt::OperatorTranspose) = opt.op

eltype{OP,SRC,DEST}(::Type{OperatorTranspose{OP,SRC,DEST}}) = eltype(OP)

# By simply switching src and dest, we implicitly identify the dual of these linear vector spaces
# with the spaces themselves.
src(opt::OperatorTranspose) = dest(operator(opt))

dest(opt::OperatorTranspose) = src(operator(opt))

is_inplace{OP,SRC,DEST}(::Type{OperatorTranspose{OP,SRC,DEST}}) = is_inplace(OP)

is_diagonal{OP,SRC,DEST}(::Type{OperatorTranspose{OP,SRC,DEST}}) = is_diagonal(OP)

# Types may implement this general transpose call to implement their transpose without creating a new operator type for it.
apply!(opt::OperatorTranspose, dest, src, coef_dest, coef_src) = apply!(opt, operator(opt), dest, src, coef_dest, coef_src)

apply!(opt::OperatorTranspose, dest, src, coef_srcdest) = apply!(opt, operator(opt), dest, src, coef_srcdest)



"An OperatorInverse represents the inverse of an operator."
immutable OperatorInverse{OP,SRC,DEST} <: AbstractOperator{SRC,DEST}
	op	::	OP

	OperatorInverse(op::AbstractOperator{DEST,SRC}) = new(op)
end

inv{SRC,DEST}(op::AbstractOperator{DEST,SRC}) = OperatorInverse{typeof(op),SRC,DEST}(op)

operator(opinv::OperatorInverse) = opinv.op

eltype{OP,SRC,DEST}(::Type{OperatorInverse{OP,SRC,DEST}}) = eltype(OP)

src(opinv::OperatorInverse) = dest(operator(opinv))

dest(opinv::OperatorInverse) = src(operator(opinv))

is_diagonal{OP,SRC,DEST}(::Type{OperatorInverse{OP,SRC,DEST}}) = is_diagonal(OP)

# Types may implement this general transpose call to implement their transpose without creating a new operator type for it.
apply!(opinv::OperatorInverse, dest, src, coef_dest, coef_src) = apply!(opinv, operator(opt), dest, src, coef_dest, coef_src)

apply!(opinv::OperatorInverse, dest, src, coef_srcdest) = apply!(opinv, operator(opt), dest, src, coef_srcdest)




"The identity operator"
immutable IdentityOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

IdentityOperator(src::FunctionSet) = IdentityOperator(src, src)

is_inplace{OP <: IdentityOperator}(::Type{OP}) = True

is_diagonal{OP <: IdentityOperator}(::Type{OP}) = True

inv(op::IdentityOperator) = IdentityOperator(dest(op), src(op))

ctranspose(op::IdentityOperator) = inv(op)

apply!(op::IdentityOperator, dest, src, coef_srcdest) = nothing

# The identity operator is, well, the identity
(*)(op2::IdentityOperator, op1::IdentityOperator) = IdentityOperator(src(op1), dest(op2))
(*)(op2::AbstractOperator, op1::IdentityOperator) = op2
(*)(op2::IdentityOperator, op1::AbstractOperator) = op1


"""
A ScalingOperator is the identity operator up to a scaling.
"""
immutable ScalingOperator{ELT,SRC} <: AbstractOperator{SRC,SRC}
	src		::	SRC
	scalar	::	ELT
end

function ScalingOperator{S <: Number,SRC}(src::SRC, scalar::S)
	ELT = promote_type(eltype(SRC), S)
	ScalingOperator{ELT,SRC}(src, scalar)
end

dest(op::ScalingOperator) = src(op)

eltype{ELT,SRC}(::Type{ScalingOperator{ELT,SRC}}) = ELT

is_inplace{OP <: ScalingOperator}(::Type{OP}) = True

is_diagonal{OP <: ScalingOperator}(::Type{OP}) = True

scalar(op::ScalingOperator) = op.scalar

ctranspose(op::ScalingOperator) = op

ctranspose{T <: Real}(op::ScalingOperator{Complex{T}}) = ScalingOperator(src(op), conj(scalar(op)))

inv(op::ScalingOperator) = ScalingOperator(src(op), 1/scalar(op))

convert{ELT,SRC}(::Type{ScalingOperator{ELT,SRC}}, op::IdentityOperator{SRC}) = ScalingOperator(src(op), ELT(1))

promote_rule{ELT,SRC}(::Type{ScalingOperator{ELT,SRC}}, ::Type{IdentityOperator{SRC}}) = ScalingOperator{ELT,SRC}

for op in (:+, :*, :-, :/)
	@eval $op(op1::ScalingOperator, op2::ScalingOperator) =
		(@assert size(op1) == size(op2); ScalingOperator(src(op1), $op(scalar(op1),scalar(op2))))
end

(*)(a::Number, op::IdentityOperator) = ScalingOperator(src(op), a)
(*)(op::IdentityOperator, a::Number) = ScalingOperator(src(op), a)

(+){SRC}(op2::IdentityOperator{SRC}, op1::IdentityOperator{SRC}) =
	convert(ScalingOperator{eltype(SRC),SRC}, op2) + convert(ScalingOperator{eltype(SRC),SRC}, op1)



function apply!(op::ScalingOperator, dest, src, coef_srcdest)
	for i in eachindex(coef_srcdest)
		coef_srcdest[i] *= op.scalar
    end
end

# Extra definition for out-of-place version to avoid making an intermediate copy
function apply!(op::ScalingOperator, dest, src, coef_dest, coef_src)
	for i in eachindex(coef_src)
		coef_dest[i] = op.scalar * coef_src[i]
	end
end



"A diagonal operator."
immutable DiagonalOperator{ELT,SRC,DEST} <: AbstractOperator{SRC,DEST}
	src		::	SRC
	dest	::	DEST
	diagonal::	Array{ELT,1}
end

diagonal(op::DiagonalOperator) = op.diagonal

is_inplace{OP <: DiagonalOperator}(::Type{OP}) = True

is_diagonal{OP <: DiagonalOperator}(::Type{OP}) = True

inv(op::DiagonalOperator) = DiagonalOperator(dest(op), src(op), op.diagonal.^(-1))

ctranspose(op::DiagonalOperator) = DiagonalOperator(dest(op), src(op), diagonal(op))

ctranspose{T <: Number}(op::DiagonalOperator{Complex{T}}) = DiagonalOperator(dest(op), src(op), conj(diagonal(op)))

function apply!(op::DiagonalOperator, dest, src, coef_srcdest)
	for i in eachindex(coef_srcdest)
		coef_srcdest[i] *= op.diagonal[i]
	end
end

# Extra definition for out-of-place version to avoid making an intermediate copy
function apply!(op::ScalingOperator, dest, src, coef_dest, coef_src)
	for i in eachindex(coef_src)
		coef_dest[i] = op.scalar * coef_src[i]
	end
end



"""
A CoefficientScalingOperator scales a single coefficient.
"""
immutable CoefficientScalingOperator{ELT,SRC} <: AbstractOperator{SRC,SRC}
	src		::	SRC
	index	::	Int
	scalar	::	ELT
end

function CoefficientScalingOperator{S <: Number, SRC}(src::SRC, index::Int, scalar::S)
	ELT = promote_type(eltype(SRC), S)
	CoefficientScalingOperator{ELT,SRC}(src, index, scalar)
end

eltype{ELT,SRC}(::Type{CoefficientScalingOperator{ELT,SRC}}) = ELT

dest(op::CoefficientScalingOperator) = src(op)

index(op::CoefficientScalingOperator) = op.index

scalar(op::CoefficientScalingOperator) = op.scalar

is_inplace{OP <: CoefficientScalingOperator}(::Type{OP}) = True

is_diagonal{OP <: CoefficientScalingOperator}(::Type{OP}) = True

ctranspose(op::CoefficientScalingOperator) = op

ctranspose{T <: Number}(op::CoefficientScalingOperator{Complex{T}}) = CoefficientScalingOperator(src(op), index(op), conj(scalar(op)))

inv(op::CoefficientScalingOperator) = CoefficientScalingOperator(src(op), index(op), 1/scalar(op))

apply!(op::CoefficientScalingOperator, dest, src, coef_srcdest) = coef_srcdest[op.index] *= op.scalar



"""
A WrappedOperator has a source and destination, as well as an embedded operator with its own
source and destination. The coefficients of the source of the WrappedOperator are passed on
unaltered as coefficients of the source of the embedded operator. The resulting coefficients
of the embedded destination are returned as coefficients of the wrapping destination.
"""
immutable WrappedOperator{OP,SRC,DEST} <: AbstractOperator{SRC,DEST}
	src		::	SRC
	dest	::	DEST

	op		::	OP
end

operator(op::WrappedOperator) = op.op

is_inplace{OP,SRC,DEST}(::Type{WrappedOperator{OP,SRC,DEST}}) = is_inplace(OP)

is_diagonal{OP,SRC,DEST}(::Type{WrappedOperator{OP,SRC,DEST}}) = is_diagonal(OP)

apply!(op::WrappedOperator, dest, src, coef_srcdest) = apply!(op.op, coef_srcdest)

apply!(op::WrappedOperator, dest, src, coef_dest, coef_src) = apply!(op.op, coef_dest, coef_src)

inv(op::WrappedOperator) = WrappedOperator(dest(op), src(op), inv(op.op))

ctranspose(op::WrappedOperator) = WrappedOperator(dest(op), src(op), ctranspose(op.op))


"""
A composite operator applies op2 after op1.
It preallocates sufficient memory to store intermediate results.
"""
immutable CompositeOperator{OP1,OP2,ELT,N,SRC,DEST} <: AbstractOperator{SRC,DEST}
	op1		::	OP1
	op2		::	OP2
	scratch	::	Array{ELT,N}	# For storing the intermediate result after applying op1

	function CompositeOperator(op1::AbstractOperator, op2::AbstractOperator)
		@assert size(op1,1) == size(op2,2)

		# Possible optimization here would be to avoid allocating memory if the second operator is in-place.
		# But even in that case, the user may invoke the operator in a non-in-place way, so let's keep it.
		new(op1, op2, zeros(ELT,size(src(op2))))
	end
end


# We could ask that DEST1 == SRC2 but that might be too strict. As long as the operators are compatible things are fine.
function CompositeOperator{SRC1,DEST1,SRC2,DEST2}(op1::AbstractOperator{SRC1,DEST1}, op2::AbstractOperator{SRC2,DEST2})
	OP1 = typeof(op1)
	OP2 = typeof(op2)
	ELT = eltype(OP1,OP2)
	CompositeOperator{OP1,OP2,ELT,length(size(src(op2))),SRC1,DEST2}(op1,op2)
end

src(op::CompositeOperator) = src(op.op1)

dest(op::CompositeOperator) = dest(op.op2)

eltype{OP1,OP2,ELT,N,SRC,DEST}(::Type{CompositeOperator{OP1,OP2,ELT,N,SRC,DEST}}) = ELT

ctranspose(op::CompositeOperator) = CompositeOperator(ctranspose(op.op2), ctranspose(op.op1))

inv(op::CompositeOperator) = CompositeOperator(inv(op.op2), inv(op.op1))

(*)(op2::AbstractOperator, op1::AbstractOperator) = CompositeOperator(op1, op2)

apply!(op::CompositeOperator, dest, src, coef_dest, coef_src) = _apply!(op, is_inplace(op.op2), coef_dest, coef_src)


function _apply!(op::CompositeOperator, op2_inplace::True, coef_dest, coef_src)
	apply!(op.op1, coef_dest, coef_src)
	apply!(op.op2, coef_dest)
end

function _apply!(op::CompositeOperator, op2_inplace::False, coef_dest, coef_src)
	apply!(op.op1, op.scratch, coef_src)
	apply!(op.op2, coef_dest, op.scratch)
end


# If it is called in-place, assuming all operators support in-place operations.
function apply!(op::CompositeOperator, dest, src, coef_srcdest)
	apply!(op.op1, coef_srcdest)
	apply!(op.op2, coef_srcdest)
end

is_inplace{OP1,OP2,ELT,N,SRC,DEST}(::Type{CompositeOperator{OP1,OP2,ELT,N,SRC,DEST}}) = is_inplace(OP1) & is_inplace(OP2)

is_diagonal{OP1,OP2,ELT,N,SRC,DEST}(::Type{CompositeOperator{OP1,OP2,ELT,N,SRC,DEST}}) = is_diagonal(OP1) & is_diagonal(OP2)


# Perhaps the parameters are excessive in this and many other types in the code. Do some tests without them later.
# A TripleCompositeOperator is implemented because it can better exploit in-place operators
# than a chain of CompositeOperator's.
immutable TripleCompositeOperator{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST} <: AbstractOperator{SRC,DEST}
	op1			::	OP1
	op2			::	OP2
	op3			::	OP3
	scratch1	::	Array{ELT,N1}	# For storing the intermediate result after applying op1
	scratch2	::	Array{ELT,N2}	# For storing the intermediate result after applying op2

	function TripleCompositeOperator(op1::AbstractOperator, op2::AbstractOperator, op3::AbstractOperator)
		@assert size(op1,1) == size(op2,2)
		@assert size(op2,1) == size(op3,2)

		new(op1, op2, op3, zeros(ELT,size(src(op2))), zeros(ELT,size(src(op3))))
	end
end

function TripleCompositeOperator{SRC1,DEST1,SRC3,DEST3}(op1::AbstractOperator{SRC1,DEST1}, op2, op3::AbstractOperator{SRC3,DEST3})
	OP1 = typeof(op1)
	OP2 = typeof(op2)
	OP3 = typeof(op3)
	ELT = eltype(OP1,OP2,OP3)
	N1 = length(size(src(op2)))
	N2 = length(size(src(op3)))
	TripleCompositeOperator{OP1,OP2,OP3,ELT,N1,N2,SRC1,DEST3}(op1, op2, op3)
end

src(op::TripleCompositeOperator) = src(op.op1)

dest(op::TripleCompositeOperator) = dest(op.op3)

eltype{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}(::Type{TripleCompositeOperator{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}}) = ELT

is_inplace{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}(::Type{TripleCompositeOperator{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}}) = is_inplace(OP1) & is_inplace(OP2) & is_inplace(OP3)

is_diagonal{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}(::Type{TripleCompositeOperator{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}}) = is_diagonal(OP1) & is_diagonal(OP2) & is_diagonal(OP3)

ctranspose(op::TripleCompositeOperator) = TripleCompositeOperator(ctranspose(op.op3), ctranspose(op.op2), ctranspose(op.op1))

inv(op::TripleCompositeOperator) = TripleCompositeOperator(inv(op.op3), inv(op.op2), inv(op.op1))

apply!(op::TripleCompositeOperator, dest, src, coef_dest, coef_src) = _apply!(op, is_inplace(op.op2), is_inplace(op.op3), coef_dest, coef_src)

function _apply!(op::TripleCompositeOperator, op2_inplace::True, op3_inplace::True, coef_dest, coef_src)
	apply!(op.op1, coef_dest, coef_src)
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
function apply!(op::TripleCompositeOperator, dest, src, coef_srcdest)
	apply!(op.op1, coef_srcdest)
	apply!(op.op2, coef_srcdest)
	apply!(op.op3, coef_srcdest)
end


(*)(op3::AbstractOperator, op2::AbstractOperator, op1::AbstractOperator) = TripleCompositeOperator(op1, op2, op3)



"A linear combination of operators: val1 * op1 + val2 * op2."
immutable OperatorSum{OP1,OP2,ELT,N,SRC,DEST} <: AbstractOperator{SRC,DEST}
	op1			::	OP1
	op2			::	OP2
	val1		::	ELT
	val2		::	ELT
	scratch		::	Array{ELT,N}

	function OperatorSum(op1::AbstractOperator, op2::AbstractOperator, val1::ELT, val2::ELT)
		# We don't enforce that source and destination of op1 and op2 are the same, but at least
		# their sizes must match.
		@assert size(src(op1)) == size(src(op2))
		@assert size(dest(op1)) == size(dest(op2))
		
		new(op1, op2, val1, val2, zeros(ELT,size(dest(op1))))
	end
end

function OperatorSum{SRC,DEST,S1 <: Number, S2 <: Number}(op1::AbstractOperator{SRC,DEST}, op2::AbstractOperator, val1::S1, val2::S2)
	ELT = promote_type(S1, S2, eltype(op1), eltype(op2))
	OperatorSum{typeof(op1), typeof(op2), ELT, index_dim(dest(op1)), SRC, DEST}(op1, op2, convert(ELT, val1), convert(ELT, val2))
end

src(op::OperatorSum) = src(op.op1)

dest(op::OperatorSum) = dest(op.op1)

eltype{OP1,OP2,ELT,N,SRC,DEST}(::Type{OperatorSum{OP1,OP2,ELT,N,SRC,DEST}}) = ELT

ctranspose(op::OperatorSum) = OperatorSum(ctranspose(op.op1), ctranspose(op.op2), conj(op.val1), conj(op.val2))


is_diagonal{OP1,OP2,ELT,N,SRC,DEST}(::Type{OperatorSum{OP1,OP2,ELT,N,SRC,DEST}}) = is_diagonal(OP1) & is_diagonal(OP2)


apply!(op::OperatorSum, dest, src, coef_srcdest) = apply!(op, op.op1, op.op2, coef_srcdest)

function apply!(op::OperatorSum, op1::AbstractOperator, op2::AbstractOperator, coef_srcdest)
	scratch = op.scratch

	apply!(op1, scratch, coef_srcdest)
	apply!(op2, coef_srcdest)

	for i in eachindex(coef_srcdest)
		coef_srcdest[i] = op.val1 * scratch[i] + op.val2 * coef_srcdest[i]
	end
end

apply!(op::OperatorSum, dest, src, coef_dest, coef_src) = apply!(op, op.op1, op.op2, coef_dest, coef_src)

function apply!(op::OperatorSum, op1::AbstractOperator, op2::AbstractOperator, coef_dest, coef_src)
	scratch = op.scratch

	apply!(op1, scratch, coef_src)
	apply!(op2, coef_dest, coef_src)

	for i in eachindex(coef_dest)
		coef_dest[i] = op.val1 * scratch[i] + op.val2 * coef_dest[i]
	end
end

# Avoid unnecessary work when one of the operators (or both) is a ScalingOperator.
function apply!(op::OperatorSum, op1::ScalingOperator, op2::ScalingOperator, coef_dest, coef_src)
	val = op.val1 * scalar(op1) + op.val2 * scalar(op2)
	for i in eachindex(coef_dest)
		coef_dest[i] = val * coef_src[i]
	end
end

function apply!(op::OperatorSum, op1::ScalingOperator, op2::AbstractOperator, coef_dest, coef_src)
	apply!(op2, coef_dest, coef_src)

	val1 = op.val1 * scalar(op1)
	for i in eachindex(coef_dest)
		coef_dest[i] = val1 * coef_src[i] + op.val2 * coef_dest[i]
	end
end

function apply!(op::OperatorSum, op1::AbstractOperator, op2::ScalingOperator, coef_dest, coef_src)
	apply!(op1, coef_dest, coef_src)

	val2 = op.val2 * scalar(op2)
	for i in eachindex(coef_dest)
		coef_dest[i] = op.val1 * coef_dest[i] + val2 * coef_src[i]
	end
end


(+)(op1::AbstractOperator, op2::AbstractOperator) = OperatorSum(op1, op2, one(eltype(op1)), one(eltype(op2)))
(-)(op1::AbstractOperator, op2::AbstractOperator) = OperatorSum(op1, op2, one(eltype(op1)), -one(eltype(op2)))



"A MatrixOperator is defined by a full matrix."
immutable MatrixOperator{ARRAY,SRC,DEST} <: AbstractOperator{SRC,DEST}
	matrix	::	ARRAY
	src		::	SRC
	dest	::	DEST

	function MatrixOperator(matrix::AbstractMatrix, src, dest)
		@assert size(matrix,1) == length(dest)
		@assert size(matrix,2) == length(src)

		new(matrix, src, dest)
	end
end


MatrixOperator{ARRAY <: AbstractMatrix,SRC,DEST}(matrix::ARRAY, src::SRC, dest::DEST) =
	MatrixOperator{ARRAY, SRC, DEST}(matrix, src, dest)

MatrixOperator{T <: Number}(matrix::AbstractMatrix{T}) = MatrixOperator(matrix, Rn{T}(size(matrix,2)), Rn{T}(size(matrix,1)))

MatrixOperator{T <: Number}(matrix::AbstractMatrix{Complex{T}}) = MatrixOperator(matrix, Cn{T}(size(matrix,2)), Cn{T}(size(matrix,1)))

eltype{ARRAY,SRC,DEST}(::Type{MatrixOperator{ARRAY,SRC,DEST}}) = eltype(ARRAY)

ctranspose(op::MatrixOperator) = MatrixOperator(ctranspose(matrix(op)), dest(op), src(op))

inv(op::MatrixOperator) = MatrixOperator(inv(matrix(op)), dest(op), src(op))

# General definition
apply!(op::MatrixOperator, dest, src, coef_dest, coef_src) = (coef_dest[:] = op.matrix * coef_src)

# Definition in terms of A_mul_B
apply!{T}(op::MatrixOperator, dest, src, coef_dest::AbstractArray{T,1}, coef_src::AbstractArray{T,1}) = A_mul_B!(coef_dest, op.matrix, coef_src)

# Be forgiving: whenever one of the coefficients is multi-dimensional, reshape to a linear array first.
apply!{T,N1,N2}(op::MatrixOperator, dest, src, coef_dest::AbstractArray{T,N1}, coef_src::AbstractArray{T,N2}) = apply!(op, reshape(coef_dest, length(coef_dest)), reshape(coef_src, length(coef_src)))


matrix(op::MatrixOperator) = op.matrix

matrix!(op::MatrixOperator, a::Array) = (a[:] = op.matrix)


# A SolverOperator wraps around a solver that is used when the SolverOperator is applied. The solver
# should implement the \ operator.
# Examples include a QR or SVD factorization, or a dense matrix.
immutable SolverOperator{Q,SRC,DEST} <: AbstractOperator{SRC,DEST}
	src		::	SRC
	dest	::	DEST
	solver	::	Q
end

apply!(op::SolverOperator, dest, src, coef_dest, coef_src) = (coef_dest[:] = op.solver \ coef_src)



