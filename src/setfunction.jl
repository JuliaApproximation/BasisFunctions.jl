# setfunction.jl

"A SetFunction represents a function from a set: it is the combination of a set and an index of that set."
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

# TODO: avoid splatting
call(f::SetFunction, x...) = call(f.set, f.idx, x...)

# For now...
# These will match with anything.
call{S <: FunctionSet{1}}(f::SetFunction{S}, x) = call(f.set, f.idx, x)
call{S <: FunctionSet{2}}(f::SetFunction{S}, x, y) = call(f.set, f.idx, x, y)
call{S <: FunctionSet{3}}(f::SetFunction{S}, x, y, z) = call(f.set, f.idx, x, y, z)
call{S <: FunctionSet{4}}(f::SetFunction{S}, x, y, z, t) = call(f.set, f.idx, x, y, z, t)


left(f::SetFunction) = left(f.set, f.idx)
right(f::SetFunction) = right(f.set, f.idx)

call!(result, f::SetFunction, grid::AbstractGrid) = call!(result, set(f), index(f), grid)


getindex(s::FunctionSet, idx) = SetFunction(s, idx)

# Multiple indices are stored as a tuple
getindex(s::FunctionSet, idx...) = SetFunction(s, idx)

