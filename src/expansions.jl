# expansions.jl

"""
A SetExpansion describes a function using its coefficient expansion in a certain function set.
Parameters:
- S is the function set.
- ELT is the numeric type of the coefficients.
- ID is the dimension of the coefficients.
"""
immutable SetExpansion{S,ELT,ID}
    set     ::  S
    coef    ::  Array{ELT,ID}

    SetExpansion(set, coef) = (@assert length(set) == length(coef); new(set,coef))
end

SetExpansion{S <: FunctionSet,ELT,ID}(set::S, coef::Array{ELT,ID}) = SetExpansion{S,ELT,ID}(set,coef)

SetExpansion(s::FunctionSet) = SetExpansion(s, eltype(s))

SetExpansion{ELT}(s::FunctionSet, ::Type{ELT}) = SetExpansion(s, zeros(ELT, size(s)))


eltype{S,ELT}(::SetExpansion{S,ELT}) = ELT
eltype{S,ELT,ID}(::Type{SetExpansion{S,ELT,ID}}) = ELT
eltype{E <: SetExpansion}(::Type{E}) = eltype(super(E))

index_dim{S,ELT,ID}(::SetExpansion{S,ELT,ID}) = ID
index_dim{S,ELT,ID}(::Type{SetExpansion{S,ELT,ID}}) = ID
index_dim{E <: SetExpansion}(::Type{E}) = index_dim(super(E))

set(e::SetExpansion) = e.set

coefficients(e::SetExpansion) = e.coef

# Delegation methods
for op in (:length, :left, :right, :grid, :numtype, :dim)
    @eval $op(e::SetExpansion) = $op(set(e))
end

# Delegation type methods
for op in (:numtype, :dim)
    @eval $op{S,ELT,ID}(::Type{SetExpansion{S,ELT,ID}}) = $op(S)
    @eval $op{E <: SetExpansion}(::Type{E}) = $op(super(E))
end

getindex(e::SetExpansion, i...) = e.coef[i...]

setindex!(e::SetExpansion, v, i...) = (e.coef[i...] = v)


call(e::SetExpansion, x...) = call_expansion(set(e), coefficients(e), x...)

call!(result, e::SetExpansion, x...) = call_expansion!(result, set(e), coefficients(e), x...)

differentiate(f::SetExpansion) = SetExpansion(set(f), differentiate(f.set, f.coef))

# This is just too cute not to do: f' is the derivative of f. Then f'' is the second derivative, and so on.
ctranspose(f::SetExpansion) = differentiate(f)


