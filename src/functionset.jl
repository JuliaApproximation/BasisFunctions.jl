# function_set.jl


######################
# Type hierarchy
######################

# Logical superset of bases and frames
abstract AbstractFunctionSet{N,T}

# A frame has more structure than a set of functions
abstract AbstractFrame{N,T} <: AbstractFunctionSet{N,T}

# A basis is a non-redundant frame
abstract AbstractBasis{N,T} <: AbstractFrame{N,T}


# Useful abstraction for special cases in 1D
typealias AbstractFunctionSet1d{T} AbstractFunctionSet{1,T}
typealias AbstractFrame1d{T} AbstractFrame{1,T}
typealias AbstractBasis1d{T} AbstractBasis{1,T}


dim{N,T}(::AbstractFunctionSet{N,T}) = N
dim{N,T}(::Type{AbstractFunctionSet{N,T}}) = N
dim{B <: AbstractFunctionSet}(::Type{B}) = dim(super(B))

numtype{N,T}(::AbstractFunctionSet{N,T}) = T
numtype{N,T}(::Type{AbstractFunctionSet{N,T}}) = T
numtype{B <: AbstractFunctionSet}(::Type{B}) = numtype(super(B))

# Function sets are real-valued by default
isreal(::AbstractFunctionSet) = True()
isreal{N,T}(::Type{AbstractFunctionSet{N,T}}) = True
isreal{B <: AbstractFunctionSet}(::Type{B}) = True

# Default element type for complex basis functions is Complex{T}
eltype(b::AbstractFunctionSet) = _eltype(b, isreal(b))
_eltype(b::AbstractFunctionSet, isreal::True) = numtype(b)
_eltype(b::AbstractFunctionSet, isreal::False) = complexify(numtype(b))

# Default dimension of the index is 1
index_dim(::AbstractFunctionSet) = 1
index_dim{N,T}(::Type{AbstractFunctionSet{N,T}}) = 1
index_dim{B <: AbstractFunctionSet}(::Type{B}) = 1

complexify{T <: Real}(::Type{T}) = Complex{T}
complexify{T <: Real}(::Type{Complex{T}}) = Complex{T}


# Is a given set a basis? In general, no, but some sets could turn out to be a basis.
# Example: a TensorProductSet that consists of a basis in each dimension.
is_basis(s::AbstractFunctionSet) = False()

# A basis is always a basis.
is_basis(b::AbstractBasis) = True()

is_frame(s::AbstractFunctionSet) = False()
is_frame(s::AbstractFrame) = True()

is_orthogonal(b::AbstractBasis) = False()
is_orthogonal{N,T}(::Type{AbstractFunctionSet{N,T}}) = False
is_orthogonal{B <: AbstractFunctionSet}(::Type{B}) = False

is_biorthogonal(b::AbstractBasis) = False()
is_biorthogonal{N,T}(::Type{AbstractFunctionSet{N,T}}) = False
is_biorthogonal{B <: AbstractFunctionSet}(::Type{B}) = False


size(s::AbstractFunctionSet) = (length(s),)

size(s::AbstractFunctionSet, j) = j==1?length(s):throw(BoundsError())


# Default set of indices: from 1 to length(s)
# Default algorithms assume this indexing for the basis functions, and the same
# linear indexing for the set of coefficients.
eachindex(s::AbstractFunctionSet) = 1:length(s)

# Default iterator over sets of functions: based on underlying index iterator.
function start(s::AbstractFunctionSet)
    iter = eachindex(s)
    (iter, start(iter))
end

function next(s::AbstractFunctionSet, state)
    iter = state[1]
    iter_state = state[2]
    idx,iter_newstate = next(iter,iter_state)
    (s[idx], (iter,iter_newstate))
end

done(s::AbstractFunctionSet, state) = done(state[1], state[2])



# Provide this implementation which Base does not include anymore
checkbounds(i::Int, j::Int) = 1 <= j <= i

checkbounds(s::AbstractFunctionSet, i) = checkbounds(length(s), i)

function checkbounds(s::AbstractFunctionSet, i1, i2)
    checkbounds(size(s,1),i1)
    checkbounds(size(s,2),i2)
end

function checkbounds(s::AbstractFunctionSet, i1, i2, i3)
    checkbounds(size(s,1),i1)
    checkbounds(size(s,2),i2)
    checkbounds(size(s,3),i3)
end

function checkbounds(s::AbstractFunctionSet, i...)
    for n = 1:length(i)
        checkbounds(size(s,n), i[n])
    end
end


gridtype(b::AbstractBasis1d) = typeof(grid(b))


support(b::AbstractBasis1d, idx) = (left(b,idx), right(b,idx))


# General vectorized calling method
#call!(b::AbstractBasis1d, idx::Int, result::AbstractArray, x::AbstractArray) = broadcast!(t -> call(b, idx, t), result, x)


# The approximation_operator function returns an operator that can be used to approximate
# a function in the function set. The operator maps a grid to a set of coefficients.
approximation_operator(s::AbstractFunctionSet) = println("Don't know how to approximate a function using a " * name(s))

# The differentation_operator function returns an operator that can be used to differentiate
# a function in the function set.
differentiation_operator(s::AbstractFunctionSet) = println("Don't know how to differentiate a function given by a " * name(s))


