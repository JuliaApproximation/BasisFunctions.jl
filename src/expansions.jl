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

    SetExpansion(set, coef) = (@assert size(set) == size(coef); new(set,coef))
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

# Delegation of methods
for op in (:length, :size, :left, :right, :grid, :numtype, :dim)
    @eval $op(e::SetExpansion) = $op(set(e))
end

# Delegation of type methods
for op in (:numtype, :dim)
    @eval $op{S,ELT,ID}(::Type{SetExpansion{S,ELT,ID}}) = $op(S)
    @eval $op{E <: SetExpansion}(::Type{E}) = $op(super(E))
end

has_basis(e::SetExpansion) = is_basis(set(e))

has_frame(e::SetExpansion) = is_frame(set(e))

getindex(e::SetExpansion, i...) = e.coef[i...]

setindex!(e::SetExpansion, v, i...) = (e.coef[i...] = v)


# This indirect call enables dispatch on the type of the set of the expansion
call(e::SetExpansion, x...) = call_set(e, set(e), coefficients(e), x...)
call_set(e::SetExpansion, s::FunctionSet, coef, x...) = call_expansion(s, coef, x...)

call(e::SetExpansion, x::Vec{2,Float64}) = call_expansion(set(e), coefficients(e), x[1], x[2])

call!(result, e::SetExpansion, x...) = call_set!(result, e, set(e), coefficients(e), x...)
call_set!(result, e::SetExpansion, s::FunctionSet, coef, x...) = call_expansion!(result, s, coef, x...)

differentiate(f::SetExpansion) = SetExpansion(set(f), differentiate(f.set, f.coef))

# This is just too cute not to do: f' is the derivative of f. Then f'' is the second derivative, and so on.
ctranspose(f::SetExpansion) = differentiate(f)


# Delegate generic operators
for op in (:extension_operator, :restriction_operator, :transform_operator)
    @eval $op(s1::SetExpansion, s2::SetExpansion) = $op(set(s1), set(s2))
end

for op in (:interpolation_operator, :evaluation_operator, :approximation_operator)
    @eval $op(s::SetExpansion) = $op(set(s))
end

differentiation_operator(s1::SetExpansion, s2::SetExpansion, var::Int...) = differentiation_operator(set(s1), set(s2), var...)
differentiation_operator(s1::SetExpansion, var::Int...) = differentiation_operator(set(s1), var...)


##############################
# Arithmetics with expansions
##############################

for op in (:+, :-)
    @eval function ($op){S,ELT1,ELT2,ID}(s1::SetExpansion{S,ELT1,ID}, s2::SetExpansion{S,ELT2,ID})
            # If the sizes are equal, we can just operate on the coefficients.
            # If not, we have to extend the smaller set to the size of the larger set.
            if size(s1) == size(s2)
                SetExpansion(set(s1), $op(coefficients(s1), coefficients(s2)))
            elseif length(s1) < s2
                # Switch the sets, so that s2 is always the smaller one
                s2,s1 = s1,s2
                s3 = extension_operator(set(s2), set(s1)) * s2
                SetExpansion(set(s1), $op(coefficients(s1), coefficients(s3)))
            end
    end
end

function apply(op::AbstractOperator, s::SetExpansion)
    @assert set(s) == src(op)

    SetExpansion(dest(op), op * coefficients(s))
end

function apply!(op::AbstractOperator, set_dest::SetExpansion, set_src::SetExpansion)
    @assert set(set_src) == src(op)
    @assert set(set_dest) == dest(op)

    apply!(op, coefficients(set_dest), coefficients(set_src))
end



