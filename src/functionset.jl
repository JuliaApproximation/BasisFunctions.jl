# functionset.jl


######################
# Type hierarchy
######################

"""
A FunctionSet is any set of functions. It is a logical superset of sets
with more structure, such as bases and frames. Each FunctionSet has a finite size,
and hence typically represents the truncation of an infinite set.
A FunctionSet has a dimension N and a numeric type T.
"""
abstract FunctionSet{N,T}

"""
An AbstractFrame has more structure than a set of functions. It is the truncation
of an infinite frame. (Since an AbstractFrame has a finite size, strictly speaking
this is usually a basis, albeit a very ill-conditioned one.)
"""
abstract AbstractFrame{N,T} <: FunctionSet{N,T}

"""
A basis is a non-redundant frame.
"""
abstract AbstractBasis{N,T} <: AbstractFrame{N,T}


# Useful abstraction for special cases in 1D
typealias FunctionSet1d{T} FunctionSet{1,T}
typealias AbstractFrame1d{T} AbstractFrame{1,T}
typealias AbstractBasis1d{T} AbstractBasis{1,T}

"The dimension of the set."
dim{N,T}(::FunctionSet{N,T}) = N
dim{N,T}(::Type{FunctionSet{N,T}}) = N
dim{B <: FunctionSet}(::Type{B}) = dim(super(B))

"The numeric type of the set."
numtype{N,T}(::FunctionSet{N,T}) = T
numtype{N,T}(::Type{FunctionSet{N,T}}) = T
numtype{B <: FunctionSet}(::Type{B}) = numtype(super(B))

"Trait to indicate whether the functions in the set are real-valued (for real arguments)."
isreal(::FunctionSet) = True()
isreal{N,T}(::Type{FunctionSet{N,T}}) = True
isreal{B <: FunctionSet}(::Type{B}) = True

"""
The eltype of a set is the typical numeric type of expansion coefficients. It is usually
either T or Complex{T}, where T is the numeric type of the set.
"""
eltype(b::FunctionSet) = _eltype(b, isreal(b))
_eltype(b::FunctionSet, isreal::True) = numtype(b)
_eltype(b::FunctionSet, isreal::False) = complexify(numtype(b))

"""
The dimension of the index of the set. This may in general be different from the dimension
of the set. For example, Fourier series on a lattice in 2D may be indexed with a single
index. Wavelets in 1D are usually indexed with two parameters, scale and position.
"""
index_dim(::FunctionSet) = 1
index_dim{N,T}(::Type{FunctionSet{N,T}}) = 1
index_dim{B <: FunctionSet}(::Type{B}) = 1

"Return a complex type associated with the argument type."
complexify{T <: Real}(::Type{T}) = Complex{T}
complexify{T <: Real}(::Type{Complex{T}}) = Complex{T}


# Is a given set a basis? In general, no, but some sets could turn out to be a basis.
# Example: a TensorProductSet that consists of a basis in each dimension.
# This is a problem that can be solved in two ways: introduce a parallel hierarchy 
# TensorProdctFrame - TensorProductBasis, or make the Basis property a trait.
# This is the trait:
is_basis(s::FunctionSet) = False()
is_basis(::Type{FunctionSet}) = False
is_basis{S <: FunctionSet}(::Type{S}) = is_basis(super(S))

is_frame(s::FunctionSet) = False()
is_frame(::Type{FunctionSet}) = False
is_frame{S <: FunctionSet}(::Type{S}) = is_frame(super(S))

# A basis is always a basis.
is_basis(b::AbstractBasis) = True()
is_basis(::Type{AbstractBasis}) = True

# And a frame is always a frame.
is_frame(s::AbstractFrame) = True()
is_frame(::Type{AbstractFrame}) = True


"Trait to indicate whether a basis is orthogonal."
is_orthogonal(b::AbstractBasis) = False()
is_orthogonal{N,T}(::Type{FunctionSet{N,T}}) = False
is_orthogonal{B <: FunctionSet}(::Type{B}) = False

is_biorthogonal(b::AbstractBasis) = False()
is_biorthogonal{N,T}(::Type{FunctionSet{N,T}}) = False
is_biorthogonal{B <: FunctionSet}(::Type{B}) = False

"Return the size of the set."
size(s::FunctionSet) = (length(s),)

"Return the size of the j-th dimension of the set (if applicable)."
size(s::FunctionSet, j) = j==1 ? length(s) : throw(BoundsError())


"""
The instantiate function takes a set type, size and numeric type as argument, and
returns an instance of the type with the given size and numeric type and using
default values for other parameters. This means the type is usually abstract.

This function is mainly used to create instances suitable for testing whether the
set type adheres to the generic interface.
"""
instantiate{B <: FunctionSet}(::Type{B}, n) = instantiate(B, n, Float64)


"Does the set implement a derivative?"
has_derivative(b::FunctionSet) = false

"Does the set have an associated interpolation grid?"
has_grid(b::FunctionSet) = false

"Does the set have an associated transform?"
has_transform(b::FunctionSet) = false

"Does the set support extension and restriction operators?"
has_extension(b::FunctionSet) = false


# A FunctionSet has logical indices and natural indices. The logical indices correspond to
# the a logical ordering of the basis functions. They correspond to the order of the coefficients
# in an expansion.
# The natural index may be closer to the mathematical definition. For example, wavelets may
# have a natural index that corresponds to the combination of scale and position. Or some
# basis functions may be defined from 0 to n, rather than from 1 to n.
# By convention, we denote a natural index variable by idxn.

"Compute the natural index corresponding to the given logical index."
natural_index(b::FunctionSet, idx) = idx

"Compute the logical index corresponding to the given natural index."
logical_index(b::FunctionSet, idxn) = idxn

# Default set of logical indices: from 1 to length(s)
# Default algorithms assume this indexing for the basis functions, and the same
# linear indexing for the set of coefficients.
# The indices may also have tensor-product structure, for tensor product sets.
eachindex(s::FunctionSet) = 1:length(s)

# Default iterator over sets of functions: based on underlying index iterator.
function start(s::FunctionSet)
    iter = eachindex(s)
    (iter, start(iter))
end

function next(s::FunctionSet, state)
    iter = state[1]
    iter_state = state[2]
    idx,iter_newstate = next(iter,iter_state)
    (s[idx], (iter,iter_newstate))
end

done(s::FunctionSet, state) = done(state[1], state[2])



# Provide this implementation which Base does not include anymore
# TODO: hook into the Julia checkbounds system, once such a thing is developed.
checkbounds(i::Int, j::Int) = (1 <= j <= i) ? nothing : throw(BoundsError())

checkbounds(s::FunctionSet, i) = checkbounds(length(s), i)

function checkbounds(s::FunctionSet, i1, i2)
    checkbounds(size(s,1),i1)
    checkbounds(size(s,2),i2)
end

function checkbounds(s::FunctionSet, i1, i2, i3)
    checkbounds(size(s,1),i1)
    checkbounds(size(s,2),i2)
    checkbounds(size(s,3),i3)
end

function checkbounds(s::FunctionSet, i...)
    for n = 1:length(i)
        checkbounds(size(s,n), i[n])
    end
end

"Return the support of the idx-th basis function."
support(b::AbstractBasis1d, idx) = (left(b,idx), right(b,idx))


function call(b::FunctionSet, i, x...)
    checkbounds(b, i)
    call_element(b, i, x...)
end


# The order of the arguments here should be different: it is more consistent to have i::Int
# before the grid. But that leads to too many ambiguity warnings...
function call(b::FunctionSet, i::Int, grid::AbstractGrid)
    result = zeros(promote_type(eltype(b),numtype(grid)), size(grid))
    call!(result, b, i, grid)
end

function call!(result, b::FunctionSet, i::Int, grid::AbstractGrid)
    @assert size(result) == size(grid)

    for k in eachindex(grid)
        result[k] = call(b, i, grid[k]...)
    end
    result
end

# This method to remove an ambiguity warning
call_expansion(b::FunctionSet, coef) = nothing

"""
Evaluate an expansion given by the set of coefficients `coef` in the point x.
"""
@generated function call_expansion{S <: Number}(b::FunctionSet, coef, xs::S...)
    xargs = [:(xs[$d]) for d = 1:length(xs)]
    quote
        T = promote_type(eltype(coef), S)
        z = zero(T)
        for i in eachindex(b)
            z = z + coef[i]*b(i, $(xargs...))
        end
        z
    end
end

@generated function call_expansion{N,T,S}(b::FunctionSet{N,T}, coef, x::Vec{N,S})
    xargs = [:(x[$d]) for d = 1:length(x)]
    quote
        call_expansion(b, coef, $(xargs...))
    end
end

# Vectorized method. Revisit once there is a standard way in Julia to treat
# vectorized function that is also fast.
# @generated to avoid splatting overhead (even though the function is vectorized,
# perhaps there is no need)
@generated function call_expansion{S <: Number}(b::FunctionSet, coef, xs::AbstractArray{S}...)
    xargs = [:(xs[$d]) for d = 1:length(xs)]
    quote
        T = promote_type(eltype(coef), S)
        result = similar(xs[1], T)
        call_expansion!(result, b, coef, $(xargs...))
    end
end

function call_expansion(b::FunctionSet, coef, grid::AbstractGrid)
    T = promote_type(eltype(coef),numtype(grid))
    result = Array(T, size(grid))
    call_expansion!(result, b, coef, grid)
end




# This function is slow - better to use transforms for special cases if available.
function call_expansion!{N}(result, b::FunctionSet{N}, coef, grid::AbstractGrid{N})
    @assert size(result) == size(grid)

    for i in eachindex(grid)
        result[i] = call_expansion(b, coef, grid[i])
    end
    result
end


@generated function call_expansion!(result, b::FunctionSet, coef, xs::AbstractArray...)
    xargs = [:(xs[$d][i]) for d = 1:length(xs)]
    quote
        for i in 1:length(xs)
            @assert size(result) == size(xs[i])
        end

        for i in eachindex(xs[1])
            result[i] = call_expansion(b, coef, $(xargs...))
        end
        result
    end
end

