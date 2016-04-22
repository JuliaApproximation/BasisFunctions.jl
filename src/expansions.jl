# expansions.jl

"""
A SetExpansion describes a function using its coefficient expansion in a certain function set.

Parameters:
- S is the function set.
- ELT is the numeric type of the coefficients.
- ID is the dimension of the coefficient matrix.
"""
immutable SetExpansion{S,ELT,ID}
    set     ::  S
    coef    ::  Array{ELT,ID}

    SetExpansion(set, coef) = (@assert size(set) == size(coef); new(set,coef))
end

SetExpansion{S <: FunctionSet,ELT,ID}(set::S, coef::Array{ELT,ID}) = SetExpansion{S,ELT,ID}(set,coef)

SetExpansion(s::FunctionSet) = SetExpansion(s, eltype(s))

SetExpansion{ELT}(s::FunctionSet, ::Type{ELT}) = SetExpansion(s, zeros(ELT, size(s)))


eltype{S,ELT,ID}(::Type{SetExpansion{S,ELT,ID}}) = ELT

index_dim{S,ELT,ID}(::Type{SetExpansion{S,ELT,ID}}) = ID
index_dim(s::SetExpansion) = index_dim(typeof(s))

set(e::SetExpansion) = e.set

coefficients(e::SetExpansion) = e.coef

# Delegation of methods
for op in (:length, :size, :left, :right, :grid, :index_dim)
    @eval $op(e::SetExpansion) = $op(set(e))
end

# Delegation of property methods
for op in (:numtype, :ndims)
    @eval $op(s::SetExpansion) = $op(set(s))
end

has_basis(e::SetExpansion) = is_basis(set(e))
has_frame(e::SetExpansion) = is_frame(set(e))

getindex(e::SetExpansion, i...) = e.coef[i...]

setindex!(e::SetExpansion, v, i...) = (e.coef[i...] = v)


# This indirect call enables dispatch on the type of the set of the expansion
call(e::SetExpansion, x...) = call_set(e, set(e), coefficients(e), promote(x...)...)
call_set(e::SetExpansion, s::FunctionSet, coef, x...) = call_expansion(s, coef, x...)

call!(result, e::SetExpansion, x...) = call_set!(result, e, set(e), coefficients(e), promote(x...)...)
call_set!(result, e::SetExpansion, s::FunctionSet, coef, x...) = call_expansion!(result, s, coef, x...)

function differentiate(f::SetExpansion, order=1)
    op = differentiation_operator(f.set, order)
    SetExpansion(dest(op), apply(op,f.coef))
end

function antidifferentiate(f::SetExpansion, order=1)
    op = antidifferentiation_operator(f.set, order)
    SetExpansion(dest(op), apply(op,f.coef))
end


# Shorthands for partial derivatives
∂x(f::SetExpansion) = differentiate(f, 1, 1)
∂y(f::SetExpansion) = differentiate(f, 2, 1)
∂z(f::SetExpansion) = differentiate(f, 3, 1)
# Shorthands for partial integrals
∫∂x(f::SetExpansion) = antidifferentiate(f, 1, 1)
∫∂y(f::SetExpansion) = antidifferentiate(f, 2, 1)
∫∂z(f::SetExpansion) = antidifferentiate(f, 3, 1)
# little helper function
ei(dim,i, coef) = tuple((coef*eye(Int,dim)[:,i])...)
# we allow the differentiation of one specific variable through the var argument
differentiate(f::SetExpansion, var, order) = differentiate(f, ei(ndims(f), var, order))
antidifferentiate(f::SetExpansion, var, order) = antidifferentiate(f, ei(ndims(f), var, order))

# To be implemented: Laplacian (needs multiplying functions)
## Δ(f::SetExpansion)

# This is just too cute not to do: f' is the derivative of f. Then f'' is the second derivative, and so on.
ctranspose(f::SetExpansion) = differentiate(f)
∫(f::SetExpansion) = antidifferentiate(f)

# Delegate generic operators
for op in (:extension_operator, :restriction_operator, :transform_operator)
    @eval $op(s1::SetExpansion, s2::SetExpansion) = $op(set(s1), set(s2))
end

for op in (:interpolation_operator, :evaluation_operator, :approximation_operator)
    @eval $op(s::SetExpansion) = $op(set(s))
end

differentiation_operator(s1::SetExpansion, s2::SetExpansion, var::Int...) = differentiation_operator(set(s1), set(s2), var...)
differentiation_operator(s1::SetExpansion, var::Int...) = differentiation_operator(set(s1), var...)

double_one{T <: Real}(::Type{T}) = one(T)
double_one{T <: Real}(::Type{Complex{T}}) = one(T) + im*one(T)

# Just generate Float64 random values and convert to the type of s
# This does not work as intended, one(Bigfloat)*rand() gives a Float64 Array
"Generate an expansion with random coefficients."
random_expansion{N,ELT}(s::FunctionSet{N,ELT}) = SetExpansion(s, double_one(ELT) * convert(Array{ELT,length(size(s))},rand(size(s))))


show(io::IO, fun::SetExpansion) = show_setexpansion(io, fun, set(fun))

function show_setexpansion(io::IO, fun::SetExpansion, fs::FunctionSet)
    println(io, "A ", ndims(fun), "-dimensional SetExpansion with ", length(coefficients(fun)), " degrees of freedom.")
    println(io, "Basis: ", name(fs))
end



##############################
# Arithmetics with expansions
##############################

for op in (:+, :-)
    @eval function ($op){S,ELT1,ELT2,ID}(s1::SetExpansion{S,ELT1,ID}, s2::SetExpansion{S,ELT2,ID})
            # If the sizes are equal, we can just operate on the coefficients.
            # If not, we have to extend the smaller set to the size of the larger set.
            if size(s1) == size(s2)
                SetExpansion(set(s1), $op(coefficients(s1), coefficients(s2)))
            elseif length(s1) < length(s2)
                s3 = extension_operator(set(s1), set(s2)) * s1
                SetExpansion(set(s2), $op(coefficients(s3), coefficients(s2)))
            else
                s3 = extension_operator(set(s2), set(s1)) * s2
                SetExpansion(set(s1), $op(coefficients(s1), coefficients(s3)))
            end
    end
end

(*)(op::AbstractOperator, e::SetExpansion) = apply(op, e)

(*)(a::Number, e::SetExpansion) = SetExpansion(set(e), a*coefficients(e))
(*)(e::SetExpansion, a::Number) = a*e

function apply(op::AbstractOperator, e::SetExpansion)
    @assert set(e) == src(op)

    SetExpansion(dest(op), op * coefficients(e))
end
