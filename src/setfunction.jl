# setfunction.jl

# A SetFunction represents a function from a set: it is the combination of a set and an index of that set.
immutable SetFunction{S <: AbstractFunctionSet,I}
    set     ::  S
    idx     ::  I   # The index can be any type, but multi-indices are stored as tuples

    function SetFunction(set, idx)
        @assert length(idx) == index_dim(set)
        new(set,idx)
    end
end

# Provide outer constructor
SetFunction{S <: AbstractFunctionSet,I}(set::S, idx::I) = SetFunction{S,I}(set,idx)

for op in (:dim, :index_dim, :numtype, :eltype)
    @eval $op(f::SetFunction) = $op(f.set)
    @eval $op{S,ID}(::Type{SetFunction{S,ID}}) = $op(S)
    @eval $op{F <: SetFunction}(::Type{F}) = $op(super(F))
end

functionset(f::SetFunction) = f.set

call{T <: Number}(b::SetFunction, x::T...) = call(b.set, b.idx, x...)

# TODO: find out how to optimize the general call so these aren't necessary
call{S <: AbstractFunctionSet{1}, T <: Number}(b::SetFunction{S}, x::T) = call(b.set, b.idx, x)
call{S <: AbstractFunctionSet{2}, T <: Number}(b::SetFunction{S}, x::T, y) = call(b.set, b.idx, x, y)
call{S <: AbstractFunctionSet{3}, T <: Number}(b::SetFunction{S}, x::T, y, z) = call(b.set, b.idx, x, y, z)
call{S <: AbstractFunctionSet{4}, T <: Number}(b::SetFunction{S}, x::T, y, z, t) = call(b.set, b.idx, x, y, z, t)


left(f::SetFunction) = left(f.set, f.idx)
right(f::SetFunction) = right(f.set, f.idx)

function call(b::SetFunction, g::AbstractGrid)
    result = Array(eltype(b), size(g))
    call!(b, result, g)
    result
end

function call!(b::SetFunction, result, grid::AbstractGrid)
    @assert size(result) == size(grid)

    for i in eachindex(grid)
        result[i] = b(grid[i])
    end
end

getindex(s::AbstractFunctionSet, idx) = SetFunction(s, idx)

# Multiple indices are stored as a tuple
getindex(s::AbstractFunctionSet, idx...) = SetFunction(s, idx)

