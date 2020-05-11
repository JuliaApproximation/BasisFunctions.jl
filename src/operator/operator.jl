
export isefficient

"""
An `AbstractOperator` is the supertype of all objects that map between function
spaces.
"""
abstract type AbstractOperator
end

dest(op::AbstractOperator) = _dest(op, dest_space(op))
_dest(op::AbstractOperator, span::Span) = dictionary(span)
_dest(op::AbstractOperator, space) = error("Generic operator does not map to the span of a dictionary.")

hasspan_dest(op::AbstractOperator) = typeof(dest_space(op)) <: Span

function apply(op::AbstractOperator, f; options...)
	if hasspan_dest(op)
		result = zeros(dest(op))
		apply!(result, op, f; options...)
	else
		error("Don't know how to apply operator $(string(op)). Please implement.")
	end
end


size(op::AbstractOperator) = _size(op, dest_space(op), src_space(op))
_size(op::AbstractOperator, dest, src) = (length(dest), length(src))

size(op::AbstractOperator, j::Int) = j <= 2 ? size(op)[j] : 1

"""
`DictionaryOperator` represents any linear operator that maps coefficients of
a source set to coefficients of a destination set. Typically, source and
destination are of type `Dictionary`.
The action of the operator is defined by providing a method for apply!.

The dimension of an operator are like a matrix: `(length(dest),length(src))`.

Source and destination should at least implement the following:
- `length`
- `size`
- `eltype`

The element type should be equal for src and dest.
"""
abstract type DictionaryOperator{T} <: AbstractOperator
end

eltype(::Type{<:DictionaryOperator{T}}) where {T} = T

# Default implementation of src and dest: assume they are fields
src(op::DictionaryOperator) = op.src
dest(op::DictionaryOperator) = op.dest

src_space(op::DictionaryOperator) = Span(src(op))
dest_space(op::DictionaryOperator) = Span(dest(op))

isreal(op::DictionaryOperator) = isreal(eltype(op)) && isreal(src(op)) && isreal(dest(op))

"""
True if the action of the operator has a better computational complexity than
the corresponding matrix-vector product.
"""
isefficient(op::DictionaryOperator) = false

size(op::DictionaryOperator) = (length(dest(op)), length(src(op)))


"Is the action of the operator in-place?"
isinplace(op::DictionaryOperator) = false

"Is the operator diagonal?"
isdiag(op::DictionaryOperator) = false

"Is the operator (knowingly) an identity operator?"
isidentity(op::DictionaryOperator) = false

function apply(op::DictionaryOperator, coef_src)
	coef_dest = zeros(eltype(op), dest(op))
	apply!(op, coef_dest, coef_src)
	coef_dest
end

# Catch applications of an operator, and do:
# - call inline implementation of operator if available
# - call apply!(op, dest(op), src(op), coef_dest, coef_src), which can be
#   implemented by operators whose action depends on src and/or dest.
function apply!(op::DictionaryOperator, coef_dest, coef_src)
	if isinplace(op)
		copyto!(coef_dest, coef_src)
		apply_inplace!(op, coef_dest)
	else
		apply_not_inplace!(op, coef_dest, coef_src)
	end
	# We expect each operator to return coef_dest, but we repeat here to make
	# sure our method is type-stable.
	coef_dest
end

# Provide a general dispatchable definition for in-place operators also
function apply!(op::DictionaryOperator, coef_srcdest)
		apply_inplace!(op, coef_srcdest)
		coef_srcdest
end

# Catch-all for missing implementations
apply_not_inplace!(op::DictionaryOperator, coef_dest, coef_src) =
	error("Operation of ", typeof(op), " on ", typeof(dest(op)), " and ", typeof(src(op)), " not implemented.")

# Catch-all for missing implementations
apply_inplace!(op::DictionaryOperator, coef_srcdest) =
	error("In-place operation of ", typeof(op), " on ", typeof(dest(op)), " and ", typeof(src(op)), " not implemented.")


"""
Apply an operator multiple times, to each column of the given argument.
"""
function apply_multiple(op::DictionaryOperator, matrix_src)
	T = eltype(op)
	matrix_dest = zeros(T, size(op,1), size(matrix_src)[2:end]...)
	coef_src = zeros(T, size(src(op)))
	coef_dest = zeros(T, size(dest(op)))
	apply_multiple!(op, matrix_dest, matrix_src, coef_dest, coef_src)
end

function apply_multiple!(op::DictionaryOperator, matrix_dest, matrix_src,
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


collect(op::DictionaryOperator) = matrix(op)

function sparse_matrix(op::DictionaryOperator; sparse_tol = 1e-14, options...)
	T = eltype(op)
	coef_src  = zeros(T, src(op))
    coef_dest = zeros(T, dest(op))
    R = spzeros(T, size(op,1), 0)
    for (i,si) in enumerate(eachindex(coef_src))
        coef_src[si] = 1
        apply!(op, coef_dest, coef_src)
        coef_src[si] = 0
        coef_dest[abs.(coef_dest).<sparse_tol] .= 0
        R = hcat(R,sparse(coef_dest))
    end
    R
end

Base.Matrix(op::DictionaryOperator) = matrix(op)

matrix(op::DictionaryOperator) =
	default_matrix(op)

function default_matrix(op::DictionaryOperator)
    a = Array{eltype(op)}(undef, size(op))
    matrix!(op, a)
end

function matrix!(op::DictionaryOperator, a)
	T = eltype(op)
    n = length(src(op))
    m = length(dest(op))

    @assert (m,n) == size(a)

    coef_src  = zeros(T, src(op))
    coef_dest = zeros(T, dest(op))

    matrix_fill!(op, a, coef_src, coef_dest)
end

function matrix_fill!(op::DictionaryOperator, a, coef_src, coef_dest)
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

function checkbounds(op::DictionaryOperator, i::Int, j::Int)
	1 <= i <= size(op,1) || throw(BoundsError())
	1 <= j <= size(op,2) || throw(BoundsError())
end

function getindex(op::DictionaryOperator, i, j)
	checkbounds(op, i, j)
	unsafe_getindex(op, i, j)
end

function unsafe_getindex(op::DictionaryOperator, i, j)
	T = eltype(op)
	coef_src = zeros(T, src(op))
	coef_dest = zeros(T, dest(op))
	coef_src[j] = one(T)
	apply!(op, coef_dest, coef_src)
	coef_dest[i]
end


"Return the diagonal of the operator."
function diag(op::DictionaryOperator)
    if isdiag(op)
        # Make data of all ones in the native representation of the operator
        all_ones = ones(src(op))
        # Apply the operator: this extracts the diagonal because the operator is diagonal
        diagonal_native = apply(op, all_ones)
        # Convert to vector
        linearize_coefficients(dest(op), diagonal_native)
    else
		# Compute the diagonal by calling unsafe_diag for each index
        [unsafe_diag(op, i) for i in 1:min(length(src(op)),length(dest(op)))]
    end
end

"Return the diagonal element op[i,i] of the operator."
function diag(op::DictionaryOperator, i)
	# Perform bounds checking and call unsafe_diag
	checkbounds(op, i, i)
	unsafe_diag(op, i)
end

# Default behaviour: call unsafe_getindex
unsafe_diag(op::DictionaryOperator, i) = unsafe_getindex(op, i, i)

# We provide a default implementation for diagonal operators
function pinv(op::DictionaryOperator, tolerance=eps(real(eltype(op))))
    @assert isdiag(op)
    newdiag = copy(diag(op))
    for i = 1:length(newdiag)
        newdiag[i] = abs(newdiag[i])>tolerance ? newdiag[i].^(-1) : 0
    end
    DiagonalOperator(dest(op),src(op), newdiag)
end

for f in (:eigvals, :svdvals, :norm, :rank)
    @eval $f(op::DictionaryOperator) = $f(Matrix(op))
end#matrix related features

function ≈(op1::DictionaryOperator,op2::DictionaryOperator; options...)
    r = rand(src(op1))
    if isapprox(op1*r,op2*r; options...)
		return true
	else
		@debug "Approx gives difference of $(norm(op1*r-op2*r))"
		return false
	end
end


"""
The function wrap_operator returns an operator with the given source and destination,
and with the action of the given operator.
"""
function wrap_operator(w_src::Dictionary, w_dest::Dictionary, op::DictionaryOperator)
    # We do some consistency checks
    @assert size(w_src) == size(src(op))
    @assert size(w_dest) == size(dest(op))
    @assert promote_type(eltype(op),operatoreltype(w_src,w_dest)) == eltype(op)
    unsafe_wrap_operator(w_src, w_dest, op)
end
