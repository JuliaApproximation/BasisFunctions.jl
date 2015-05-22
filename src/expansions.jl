# expansions.jl


immutable SetExpansion{S,ELT,ID}
    set     ::  S
    coef    ::  Array{ELT,ID}
end

SetExpansion{S <: AbstractFunctionSet}(s::S) = SetExpansion(s, eltype(s))

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

length(e::SetExpansion) = length(function_set(e))

left(e::SetExpansion) = left(function_set(e))

right(e::SetExpansion) = right(function_set(e))

getindex(e::SetExpansion, i...) = e.coef[i...]

setindex!(e::SetExpansion, v, i...) = (e.coef[i...] = v)

grid(e::SetExpansion) = grid(function_set(e))


expansion(s::AbstractFunctionSet) = SetExpansion(s)

expansion(s::AbstractFunctionSet, coef) = SetExpansion(s, coef)

expansion(s::TensorProductBasis) = TensorProductExpansion(s)

expansion{T}(s::TensorProductBasis, coef::Array{T,1}) = TensorProductExpansion(s, reshape(coef, size(s)))

expansion{T,N}(s::TensorProductBasis, coef::Array{T,N}) = TensorProductExpansion(s, coef)


#call{T <: Number,S}(e::SetExpansion{S}, x::T, y...) = call_compactsupport(has_compact_support(S), e, x, y...)
#
# function call_compactsupport{T <: Number}(::Type{True}, e::SetExpansion, x::T, y...)
#   z = zero(T)
#   I = active_indices(e.set, x, y...)
#   for i = 1:length(I)
#       idx = I[i]
#       z = z + e.coef[idx] * call(e.basis, idx, x, y...)
#   end
#   z
#end
#
#function call_compactsupport{T <: Number}(::Type{False}, e::SetExpansion, x::T, y...)
#   z = zero(T)
#   for idx = 1:length(e.set)
#       z = z + e.coef[idx] * call(e.set, idx, x, y...)
#   end
#   z
#end

function call{T <: Number}(e::SetExpansion, x::T...)
    z = zero(T)
    for idx = 1:length(e.set)
        z = z + e.coef[idx] * call(e.set, idx, x...)
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



