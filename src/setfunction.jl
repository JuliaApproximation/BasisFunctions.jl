# setfunction.jl

# A SetFunction represents a function from a set: it is the combination of a set and an index of that set.
immutable SetFunction{S <: FunctionSet,I}
    set     ::  S
    idx     ::  I   # The index can be any type, but multi-indices are stored as tuples

    function SetFunction(set, idx)
        @assert length(idx) == index_dim(set)
        new(set,idx)
    end
end

# Provide outer constructor
SetFunction{S <: FunctionSet,I}(set::S, idx::I) = SetFunction{S,I}(set,idx)

index(f::SetFunction) = f.idx

for op in (:dim, :index_dim, :numtype, :eltype)
    @eval $op(f::SetFunction) = $op(f.set)
    @eval $op{S,ID}(::Type{SetFunction{S,ID}}) = $op(S)
    @eval $op{F <: SetFunction}(::Type{F}) = $op(super(F))
end

functionset(f::SetFunction) = f.set

call{T <: Number}(f::SetFunction, x::T...) = call(f.set, f.idx, x...)

# TODO: find out how to optimize the general call so these aren't necessary
call{S <: FunctionSet{1}, T <: Number}(f::SetFunction{S}, x::T) = call(f.set, f.idx, x)
call{S <: FunctionSet{2}, T <: Number}(f::SetFunction{S}, x::T, y) = call(f.set, f.idx, x, y)
call{S <: FunctionSet{3}, T <: Number}(f::SetFunction{S}, x::T, y, z) = call(f.set, f.idx, x, y, z)
call{S <: FunctionSet{4}, T <: Number}(f::SetFunction{S}, x::T, y, z, t) = call(f.set, f.idx, x, y, z, t)


left(f::SetFunction) = left(f.set, f.idx)
right(f::SetFunction) = right(f.set, f.idx)

call(f::SetFunction, grid::AbstractGrid) = call(set(f), grid, index(f))

call!(result, f::SetFunction, grid::AbstractGrid) = call!(result, set(f), grid, index(f))


getindex(s::FunctionSet, idx) = SetFunction(s, idx)

# Multiple indices are stored as a tuple
getindex(s::FunctionSet, idx...) = SetFunction(s, idx)

