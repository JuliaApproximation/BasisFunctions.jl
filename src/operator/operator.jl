# operator.jl

"""
A `GenericOperator` is the supertype of all objects that map between function
spaces.
"""
abstract type GenericOperator
end


"""
`AbstractOperator` represents any linear operator that maps coefficients of
a source set to coefficients of a destination set. Typically, source and
destination are of type `Span`.
The action of the operator is defined by providing a method for apply!.

The dimension of an operator are like a matrix: `(length(dest),length(src))`.

Source and destination should at least implement the following:
- `length`
- `size`
- `eltype`

The element type should be equal for src and dest.
"""
abstract type AbstractOperator{T} <: GenericOperator
end

eltype(::AbstractOperator{T}) where {T} = T
eltype(::Type{AbstractOperator{T}}) where {T} = T
eltype(::Type{OP}) where {OP <: AbstractOperator} = eltype(supertype(OP))

# Default implementation of src and dest: assume they are fields
src(op::AbstractOperator) = op.src
dest(op::AbstractOperator) = op.dest

isreal(op::AbstractOperator) = isreal(src(op)) && isreal(dest(op))

"Return a suitable element type for an operator between the given spans."
op_eltype(src::Span, dest::Span) = _op_eltype(coeftype(src), coeftype(dest))
_op_eltype(::Type{T}, ::Type{T}) where {T <: Number} = T
_op_eltype(::Type{T}, ::Type{S}) where {T <: Number, S <: Number} = promote_type(T,S)
_op_eltype(::Type{SVector{N,T}}, ::Type{SVector{M,S}}) where {M,N,S,T} = SMatrix{M,N,promote_type(T,S)}
_op_eltype(::Type{T}, ::Type{S}) where {T,S} = promote_type(T,S)

"""
Return suitably promoted types such that `D = A*S` are the types of the multiplication.
"""
op_eltypes(src::Span, dest::Span, T = op_eltype(src, dest)) = _op_eltypes(coeftype(src), coeftype(dest), T)
_op_eltypes(::Type{S}, ::Type{D}, ::Type{A}) where {S <: Number, D <: Number, A <: Number} =
	(promote_type(S, D, A), promote_type(S, D, A), promote_type(S, D, A))
_op_eltypes(::Type{SVector{N,S}}, ::Type{SVector{M,D}}, ::Type{SMatrix{M,N,A}}) where {N,M,S <: Number, D <: Number, A <: Number} =
	(SVector{N,S}, SVector{M, promote_type(S, D, A)}, SMatrix{M,N,promote_type(S, D, A)})


"Promote the element type of the given operator."
promote_eltype(op::AbstractOperator{T}, ::Type{T}) where {T} = op
promote_eltype(op::AbstractOperator{T}, ::Type{S}) where {T,S} =
	similar_operator(op, S, src(op), dest(op))

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
	coef_dest = zeros(dest(op))
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
		# We pass on the sets, rather than the spans, because the coefficient
		# type is implicit in the coefficients
		apply!(op, dictionary(dest(op)), dictionary(src(op)), coef_dest, coef_src)
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

(*)(op::AbstractOperator, coef_src) = apply(op, coef_src)

"""
Apply an operator multiple times, to each column of the given argument.
"""
function apply_multiple(op::AbstractOperator, matrix_src)
	ELT = eltype(op)
	matrix_dest = zeros(ELT, size(op,1), size(matrix_src)[2:end]...)
	coef_src = zeros(ELT, size(src(op)))
	coef_dest = zeros(ELT, size(dest(op)))
	apply_multiple!(op, matrix_dest, matrix_src, coef_dest, coef_src)
end

function apply_multiple!(op::AbstractOperator, matrix_dest, matrix_src,
	coef_dest = zeros(eltype(matrix_dest), size(dest(op))),
	coef_src  = zeros(eltype(matrix_src), size(src(op))))

	# Make sure the first dimensions of the matrices agree with the dimensions
	# of the operator
	@assert size(matrix_dest, 1) == size(op,1)
	@assert size(matrix_src, 1) == size(op,2)
	# and that the remaining dimensions agree with each other
	for i in 2:ndims(matrix_dest)
		@assert size(matrix_src,i) == size(matrix_dest,i)
	end
	for i in 1:size(matrix_src,2)
		for j in eachindex(coef_src)
			coef_src[j] = matrix_src[j,i]
		end
		apply!(op, coef_dest, coef_src)
		for j in eachindex(coef_dest)
			matrix_dest[j,i] = coef_dest[j]
		end
	end
	matrix_dest
end


collect(op::AbstractOperator) = matrix(op)

function sparse_matrix(op::AbstractOperator;sparse_tol = 1e-14, options...)
	coef_src  = zeros(src(op))
    coef_dest = zeros(dest(op))
    R = spzeros(eltype(op),size(op,1),0)
    for (i,si) in enumerate(eachindex(coef_src))
        coef_src[si] = 1
        apply!(op, coef_dest, coef_src)
        coef_src[si] = 0
        coef_dest[abs.(coef_dest).<sparse_tol] = 0
        R = hcat(R,sparse(coef_dest))
    end
    R
end


function matrix(op::AbstractOperator)
    a = Array{eltype(op)}(size(op))
    matrix!(op, a)
end

function matrix!(op::AbstractOperator, a)
    n = length(src(op))
    m = length(dest(op))

    @assert (m,n) == size(a)

    coef_src  = zeros(src(op))
    coef_dest = zeros(dest(op))

    matrix_fill!(op, a, coef_src, coef_dest)
end

function matrix_fill!(op::AbstractOperator, a, coef_src, coef_dest)
	T = eltype(op)
    for (i,si) in enumerate(eachindex(coef_src))
		coef_src[si] = one(T)
        apply!(op, coef_dest, coef_src)
		coef_src[si] = zero(T)
        for (j,dj) in enumerate(eachindex(coef_dest))
            a[j,i] = coef_dest[dj]
        end
    end
    a
end

function checkbounds(op::AbstractOperator, i::Int, j::Int)
	1 <= i <= size(op,1) || throw(BoundsError())
	1 <= j <= size(op,2) || throw(BoundsError())
end

function getindex(op::AbstractOperator, i, j)
	checkbounds(op, i, j)
	unsafe_getindex(op, i, j)
end

function unsafe_getindex(op::AbstractOperator, i, j)
	coef_src = zeros(src(op))
	coef_dest = zeros(dest(op))
	coef_src[j] = one(eltype(op))
	apply!(op, coef_dest, coef_src)
	coef_dest[i]
end

"Return the diagonal of the operator."
function diagonal(op::AbstractOperator)
    if is_diagonal(op)
        # Make data of all ones in the native representation of the operator
        all_ones = ones(src(op))
        # Apply the operator: this extracts the diagonal because the operator is diagonal
        diagonal_native = apply(op, all_ones)
        # Convert to vector
        linearize_coefficients(dest(op), diagonal_native)
    else
		# Compute the diagonal by calling unsafe_diagonal for each index
        [unsafe_diagonal(op, i) for i in 1:min(length(src(op)),length(dest(op)))]
    end
end

"Return the diagonal element op[i,i] of the operator."
function diagonal(op::AbstractOperator, i)
	# Perform bounds checking and call unsafe_diagonal
	checkbounds(op, i, i)
	unsafe_diagonal(op, i)
end

# Default behaviour: call unsafe_getindex
unsafe_diagonal(op::AbstractOperator, i) = unsafe_getindex(op, i, i)


function inv_diagonal(op::AbstractOperator)
    @assert is_diagonal(op)
    d = diagonal(op)
    # Avoid getting Inf values, we prefer a pseudo-inverse in this case
    d[find(d.==0)] = Inf
    DiagonalOperator(dest(op), src(op), d.^(-1))
end
