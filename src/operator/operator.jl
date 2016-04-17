# operator.jl


"""
AbstractOperator represents any linear operator that maps coefficients of
a source set to coefficients of a destination set. Typically, source and
destination are of type FunctionSet.
The action of the operator is defined by providing a method for apply!.

The dimension of an operator are like a matrix: (length(dest),length(src)).

Source and destination should at least implement the following:
- length
- size
- eltype

The element type (eltype) should be equal for src and dest.
"""
abstract AbstractOperator{ELT}

eltype{ELT}(::Type{AbstractOperator{ELT}}) = ELT
eltype{OP <: AbstractOperator}(::Type{OP}) = eltype(super(OP))

op_eltype(src::FunctionSet, dest::FunctionSet) = promote_type(eltype(src),eltype(dest))

# Default implementation of src and dest
src(op::AbstractOperator) = op.src
dest(op::AbstractOperator) = op.dest

# The size of the operator as a linear map from source to destination.
# It is equal to the size of its matrix representation.
size(op::AbstractOperator) = (length(dest(op)), length(src(op)))

size(op::AbstractOperator, j::Int) = j==1 ? length(dest(op)) : length(src(op))

#+(op1::AbstractOperator, op2::AbstractOperator) = +(promote(op1,op2)...)

"Is the action of the operator in-place?"
is_inplace(op::AbstractOperator) = false

"Is the operator diagonal?"
is_diagonal(op::AbstractOperator) = false


function apply(op::AbstractOperator, coef_src)
		coef_dest = Array(eltype(op), size(dest(op)))
		apply!(op, coef_dest, coef_src)
		coef_dest
end

# Catch applications of an operator, and do:
# - call inline implementation of operator if available
# - call apply!(op, dest(op), src(op), coef_dest, coef_src), which can be
#   implemented by operators whose action depends on src and/or dest.
function apply!(op::AbstractOperator, coef_dest, coef_src)
	if is_inplace(op)
		copy!(coef_dest, coef_src)
		apply_inplace!(op, coef_dest)
	else
		apply!(op, dest(op), src(op), coef_dest, coef_src)
	end
	# We expect each operator to return coef_dest, but we repeat here to make
	# sure our method is type-stable.
	coef_dest
end

# Provide a general dispatchable definition for in-place operators also
function apply!(op::AbstractOperator, coef_srcdest)
		apply_inplace!(op, coef_srcdest)
		coef_srcdest
end

function apply_inplace!(op::AbstractOperator, coef_srcdest)
		apply_inplace!(op, dest(op), src(op), coef_srcdest)
		coef_srcdest
end


# Catch-all for missing implementations
function apply!(op::AbstractOperator, dest, src, coef_dest, coef_src)
	println("Operation of ", typeof(op), " on ", typeof(dest), " and ", typeof(src), " not implemented.")
	throw(InexactError())
end

# Catch-all for missing implementations
function apply_inplace!(op::AbstractOperator, dest, src, coef_srcdest)
	println("In-place operation of ", typeof(op), " not implemented.")
	throw(InexactError())
end

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
immutable OperatorTranspose{OP,ELT} <: AbstractOperator{ELT}
		op	::	OP

		OperatorTranspose(op::AbstractOperator{ELT}) = new(op)
end

OperatorTranspose(op::AbstractOperator) =
	OperatorTranspose{typeof(op),eltype(op)}(op)

ctranspose(op::AbstractOperator) = OperatorTranspose(op)

operator(opt::OperatorTranspose) = opt.op

# By simply switching src and dest, we implicitly identify the dual of these linear vector spaces
# with the spaces themselves.
src(opt::OperatorTranspose) = dest(operator(opt))

dest(opt::OperatorTranspose) = src(operator(opt))

for property in [:is_inplace, :is_diagonal]
	@eval $property(opt::OperatorTranspose) = $property(operator(opt))
end

# Types may implement this general transpose call to implement their transpose without creating a new operator type for it.
apply!(opt::OperatorTranspose, dest, src, coef_dest, coef_src) =
	apply_transpose!(operator(opt), src, dest, coef_dest, coef_src)

apply_inplace!(opt::OperatorTranspose, dest, src, coef_srcdest) =
	apply_transpose_inplace!(operator(opt), src, dest, coef_srcdest)



"An OperatorInverse represents the inverse of an operator."
immutable OperatorInverse{OP,ELT} <: AbstractOperator{ELT}
		op	::	OP

		OperatorInverse(op::AbstractOperator{ELT}) = new(op)
end

OperatorInverse(op::AbstractOperator) =
	OperatorInverse{typeof(op),eltype(op)}(op)

inv(op::AbstractOperator) = OperatorInverse(op)

operator(op::OperatorInverse) = op.op

src(op::OperatorInverse) = dest(operator(op))

dest(op::OperatorInverse) = src(operator(op))

for property in [:is_inplace, :is_diagonal]
	@eval $property(op::OperatorInverse) = $property(operator(op))
end

# Types may implement this general transpose call to implement their transpose without creating a new operator type for it.
apply!(op::OperatorInverse, dest, src, coef_dest, coef_src) =
	apply_inv!(operator(op), src, dest, coef_dest, coef_src)

apply!(op::OperatorInverse, dest, src, coef_srcdest) =
	apply_inv_inplace!(operator(op), src, dest, coef_srcdest)



include("composite_operator.jl")

include("special_operators.jl")
