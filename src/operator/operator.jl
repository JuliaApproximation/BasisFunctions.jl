# operator.jl


"""
AbstractOperator represents any linear operator that maps SRC to DEST.
Typically, SRC and DEST are of type FunctionSet, but that is not enforced.
The action of the operator is defined by providing a method for apply!.

The dimension of an operator are like a matrix: (length(dest),length(src)).

SRC and DEST should at least implement the following:
- length
- size
- numtype
- eltype

The element type (eltype) should be equal for SRC and DEST.
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
    # These assertions prevent a lot of inlining:
	## @assert length(coef_dest) == length(dest(op))
	## @assert length(coef_src) == length(src(op))
    ## @assert eltype(op) == eltype(coef_dest)
    ## @assert eltype(op) == eltype(coef_src)
    
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
function apply!(op::AbstractOperator, coef_srcdest)
	## @assert size(dest(op)) == size(src(op))
	apply!(op, dest(op), src(op), coef_srcdest)
end

# Catch-all for missing implementations
apply!(op::AbstractOperator, dest, src, coef_dest, coef_src) = println("Operation of ", op, " not implemented.")

# Catch-all for missing implementations
apply!(op::AbstractOperator, dest, src, coef_srcdest) = println("In-place operation of ", op, " not implemented.")

(*)(op::AbstractOperator, coef_src::AbstractArray) = apply(op, coef_src)


collect(op::AbstractOperator) = matrix(op)

function matrix(op::AbstractOperator)
    a = Array(eltype(op), size(op))
    matrix!(op, a)
end

function matrix!{T}(op::AbstractOperator, a::AbstractArray{T})
    n = length(src(op))
    m = length(dest(op))
    
    @assert (m,n) == size(a)
    
    r = zeros(T, size(src(op)))
    s = zeros(T, size(dest(op)))
    matrix_fill!(op, a, r, s)
end

function matrix_fill!(op::AbstractOperator, a, r, s)
    for i = 1:length(r)
        if (i > 1)
            r[i-1] = 0
        end
        r[i] = 1
        apply!(op, s, r)
        for j = 1:length(s)
            a[j,i] = s[j]
        end
    end
    a
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
apply!(opinv::OperatorInverse, dest, src, coef_dest, coef_src) = apply!(opinv, operator(opinv), dest, src, coef_dest, coef_src)

apply!(opinv::OperatorInverse, dest, src, coef_srcdest) = apply!(opinv, operator(opinv), dest, src, coef_srcdest)



include("composite_operator.jl")

include("special_operators.jl")

