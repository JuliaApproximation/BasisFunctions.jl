# basis_types.jl


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
isreal(::AbstractFunctionSet) = True
isreal{N,T}(::Type{AbstractFunctionSet{N,T}}) = True
isreal{B <: AbstractFunctionSet}(::Type{B}) = True

# Default element type for complex basis functions is Complex{T}
eltype(b::AbstractFunctionSet) = _eltype(b, isreal(b))
_eltype(b::AbstractFunctionSet, ::Type{True}) = numtype(b)
_eltype(b::AbstractFunctionSet, ::Type{False}) = complexify(numtype(b))

# Default dimension of the index is 1
index_dim(::AbstractFunctionSet) = 1
index_dim{N,T}(::Type{AbstractFunctionSet{N,T}}) = 1
index_dim{B <: AbstractFunctionSet}(::Type{B}) = 1


is_orthogonal(b::AbstractBasis) = False
is_biorthogonal(b::AbstractBasis) = False


size(s::AbstractFunctionSet) = (length(s),)

size(s::AbstractFunctionSet, j) = j==1?length(s):throw(BoundsError())


# Iterate over sets of functions
start(s::AbstractFunctionSet) = 1
done(s::AbstractFunctionSet, i::Int) = (length(s) < i)
next(s::AbstractFunctionSet, i::Int) = (s[i], i+1)

eachindex(s::AbstractFunctionSet) = 1:length(s)

stagedfunction eachindex{T,N}(s::AbstractFunctionSet{T,N})
    startargs = fill(1, N)
    stopargs = [:(size(s,$i)) for i=1:N]
    :(CartesianRange(CartesianIndex{$N}($(startargs...)), CartesianIndex{$N}($(stopargs...))))
end


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


gridtype(b::AbstractBasis1d) = typeof(natural_grid(b))


support(b::AbstractBasis1d, idx) = (left(b,idx), right(b,idx))

# Waypoints are points of discontinuity of the basis functions, such that the basis functions are smooth in between two consecutive waypoints.
waypoints(b::AbstractBasis1d, idx) = (left(b,idx), right(b,idx))



function overlap(b::AbstractBasis1d, idx1, idx2)
    support1 = support(b, idx1)
    support2 = support(b, idx2)
    ~((support1[2] <= support2[1]) || (support1[1] >= support2[2]))
end

has_compact_support{B <: AbstractBasis1d}(::Type{B}) = False

# General vectorized calling method
call!(b::AbstractBasis1d, idx::Int, result::AbstractArray, x::AbstractArray) = broadcast!(t -> call(b, idx, t), result, x)




immutable SetFunction{S <: AbstractFunctionSet,ID}
    set     ::  S
    idx     ::  NTuple{ID,Int}
end

dim(f::SetFunction) = dim(f.set)
dim{S,ID}(::Type{SetFunction{S,ID}}) = dim(S)
dim{F <: SetFunction}(::Type{F}) = dim(super(F))

index_dim{S,ID}(::SetFunction{S,ID}) = ID
index_dim{S,ID}(::Type{SetFunction{S,ID}}) = ID
index_dim{F <: SetFunction}(::Type{F}) = index_dim(super(F))

eltype(f::SetFunction) = eltype(f.set)
eltype{S,ID}(::Type{SetFunction{S,ID}}) = eltype(S)
eltype{F <: SetFunction}(::Type{F}) = eltype(super(F))

functionset(f::SetFunction) = f.set

call{S <: AbstractFunctionSet{1}, T <: Number}(b::SetFunction{S}, x::T) = call(b.set, b.idx, x)
call{S <: AbstractFunctionSet{2}, T <: Number}(b::SetFunction{S}, x::T, y) = call(b.set, b.idx, x, y)
call{S <: AbstractFunctionSet{3}, T <: Number}(b::SetFunction{S}, x::T, y, z) = call(b.set, b.idx, x, y, z)
call{S <: AbstractFunctionSet{4}, T <: Number}(b::SetFunction{S}, x::T, y, z, t) = call(b.set, b.idx, x, y, z, t)

call(b::SetFunction, x...) = call(b.set, b.idx, x...)

left(f::SetFunction) = left(f.set, f.idx)
right(f::SetFunction) = right(f.set, f.idx)

function call(b::SetFunction, g::AbstractGrid)
    result = Array(eltype(b), size(g))
    call!(b, result, g)
    result
end

function call!(b::SetFunction, result, g::AbstractGrid)
    for i in eachindex(g)
        result[i] = b(g[i])
    end
end

getindex(s::AbstractFunctionSet, idx...) = (checkbounds(s, idx...); SetFunction(s, idx))




