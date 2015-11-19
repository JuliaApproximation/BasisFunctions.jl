# functionset.jl


######################
# Type hierarchy
######################

# Logical superset of bases and frames
abstract FunctionSet{N,T}

# A frame has more structure than a set of functions
abstract AbstractFrame{N,T} <: FunctionSet{N,T}

# A basis is a non-redundant frame
abstract AbstractBasis{N,T} <: AbstractFrame{N,T}


# Useful abstraction for special cases in 1D
typealias FunctionSet1d{T} FunctionSet{1,T}
typealias AbstractFrame1d{T} AbstractFrame{1,T}
typealias AbstractBasis1d{T} AbstractBasis{1,T}


dim{N,T}(::FunctionSet{N,T}) = N
dim{N,T}(::Type{FunctionSet{N,T}}) = N
dim{B <: FunctionSet}(::Type{B}) = dim(super(B))

numtype{N,T}(::FunctionSet{N,T}) = T
numtype{N,T}(::Type{FunctionSet{N,T}}) = T
numtype{B <: FunctionSet}(::Type{B}) = numtype(super(B))

# Function sets are real-valued by default
isreal(::FunctionSet) = True()
isreal{N,T}(::Type{FunctionSet{N,T}}) = True
isreal{B <: FunctionSet}(::Type{B}) = True

# Default element type for complex basis functions is Complex{T}
eltype(b::FunctionSet) = _eltype(b, isreal(b))
_eltype(b::FunctionSet, isreal::True) = numtype(b)
_eltype(b::FunctionSet, isreal::False) = complexify(numtype(b))

# Default dimension of the index is 1
index_dim(::FunctionSet) = 1
index_dim{N,T}(::Type{FunctionSet{N,T}}) = 1
index_dim{B <: FunctionSet}(::Type{B}) = 1

complexify{T <: Real}(::Type{T}) = Complex{T}
complexify{T <: Real}(::Type{Complex{T}}) = Complex{T}


# Is a given set a basis? In general, no, but some sets could turn out to be a basis.
# Example: a TensorProductSet that consists of a basis in each dimension.
is_basis(s::FunctionSet) = False()

# A basis is always a basis.
is_basis(b::AbstractBasis) = True()

is_frame(s::FunctionSet) = False()
is_frame(s::AbstractFrame) = True()

is_orthogonal(b::AbstractBasis) = False()
is_orthogonal{N,T}(::Type{FunctionSet{N,T}}) = False
is_orthogonal{B <: FunctionSet}(::Type{B}) = False

is_biorthogonal(b::AbstractBasis) = False()
is_biorthogonal{N,T}(::Type{FunctionSet{N,T}}) = False
is_biorthogonal{B <: FunctionSet}(::Type{B}) = False


size(s::FunctionSet) = (length(s),)

size(s::FunctionSet, j) = j==1?length(s):throw(BoundsError())


# Default set of indices: from 1 to length(s)
# Default algorithms assume this indexing for the basis functions, and the same
# linear indexing for the set of coefficients.
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
checkbounds(i::Int, j::Int) = 1 <= j <= i

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


support(b::AbstractBasis1d, idx) = (left(b,idx), right(b,idx))


function call(b::FunctionSet, grid::AbstractGrid, i::Int)
    result = zeros(eltype(b), size(grid))
    call!(result, b, grid, i)
    result
end

function call!(result, b::FunctionSet, grid::AbstractGrid, i::Int)
    @assert size(result) == size(grid)

    for i in eachindex(grid)
        result[i] = call(b, i, grid[i]...)
    end
end


"""
Evaluate an expansion given by the set of coefficients `coef` in the point x.
"""
function call_expansion(b::FunctionSet, coef, x...)
    z = zero(eltype(coef))
    for i in eachindex(b)
        z = z + coef[i]*b(i, x...)
    end
    z
end


function call_expansion{T <: Number}(b::FunctionSet, coef, x::AbstractArray{T})
    result = similar(x, eltype(coef))
    call!(result, b, coef, x)
    result
end

function call_expansion(b::FunctionSet, coef, g::AbstractGrid)
    result = Array(eltype(coef), size(g))
    call_expansion!(result, b, coef, g)
    result
end


function call_expansion!(result, b::FunctionSet, coef, g::AbstractGrid)
    @assert size(result) == size(g)

    for i in eachindex(g)
        result[i] = call_expansion(b, coef, g[i]...)
    end
end
