# expansions.jl


immutable SetExpansion{S,ELT,ID}
    set     ::  S
    coef    ::  Array{ELT,ID}

    SetExpansion(set, coef) = (@assert length(set) == length(coef); new(set,coef))
end

SetExpansion{S <: AbstractFunctionSet,ELT,ID}(set::S, coef::Array{ELT,ID}) = SetExpansion{S,ELT,ID}(set,coef)

SetExpansion(s::AbstractFunctionSet) = SetExpansion(s, eltype(s))

SetExpansion{ELT}(s::AbstractFunctionSet, ::Type{ELT}) = SetExpansion(s, zeros(ELT, size(s)))


numtype(e::SetExpansion) = numtype(e.set)
numtype{S,ELT,ID}(::Type{SetExpansion{S,ELT,ID}}) = numtype(S)
numtype{E <: SetExpansion}(::Type{E}) = numtype(super(E))

eltype{S,ELT}(::SetExpansion{S,ELT}) = ELT
eltype{S,ELT,ID}(::Type{SetExpansion{S,ELT,ID}}) = ELT
eltype{E <: SetExpansion}(::Type{E}) = eltype(super(E))

dim(e::SetExpansion) = dim(set(e))
dim{S,ELT,ID}(::Type{SetExpansion{S,ELT,ID}}) = dim(S)
dim{E <: SetExpansion}(::Type{E}) = dim(super(E))

index_dim{S,ELT,ID}(::SetExpansion{S,ELT,ID}) = ID
index_dim{S,ELT,ID}(::Type{SetExpansion{S,ELT,ID}}) = ID
index_dim{E <: SetExpansion}(::Type{E}) = index_dim(super(E))

set(e::SetExpansion) = e.set

coefficients(e::SetExpansion) = e.coef

length(e::SetExpansion) = length(set(e))

left(e::SetExpansion) = left(set(e))

right(e::SetExpansion) = right(set(e))

getindex(e::SetExpansion, i...) = e.coef[i...]

setindex!(e::SetExpansion, v, i...) = (e.coef[i...] = v)

grid(e::SetExpansion) = grid(set(e))


expansion(s::AbstractFunctionSet) = SetExpansion(s)

expansion(s::AbstractFunctionSet, coef) = SetExpansion(s, coef)



function call{T <: Number}(e::SetExpansion, x::T...)
    z = zero(T)
    for idx in eachindex(set(e))
        z = z + e.coef[idx] * call(set(e), idx, x...)
    end
    z
end

function call{T <: Number}(e::SetExpansion, x::AbstractArray{T})
    result = similar(x, eltype(e))
    call!(e, result, x)
    result
end

function call(e::SetExpansion, g::AbstractGrid)
    result = Array(eltype(e), size(g))
    call!(e, result, g)
    result
end

function call!{N}(e::SetExpansion, result::AbstractArray, g::AbstractGrid{N})
    x = Array(eltype(e), N)
    for i in eachindex(g)
        getindex!(g, x, i)
        result[i] = call(e, x...)
    end
end

function call!(e::SetExpansion, result::AbstractArray, x::AbstractArray)
    @assert size(result) == size(x)
    for i = 1:length(x)
        result[i] = call(e, x[i])
    end
end



