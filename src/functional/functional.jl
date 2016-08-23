# functional.jl


"""
AbstractFunctional is the supertype of all functionals in BasisFunctions.
Any functional has a source and an element type.

The dimensions of a functional are like a row: (1,length(src)).
"""
abstract AbstractFunctional{ELT}

# Assume that src is a field of f. Override this routine if it isn't.
src(f::AbstractFunctional) = f.src

eltype{ELT}(::Type{AbstractFunctional{ELT}}) = ELT
eltype{F <: AbstractFunctional}(::Type{F}) = eltype(supertype(F))

length(f::AbstractFunctional) = length(src(f))

size(f::AbstractFunctional) = (1,length(f))

apply(f::AbstractFunctional, coef) = apply_functional(f, src(f), coef)

(*)(f::AbstractFunctional, coef::AbstractArray) = apply(f, coef)


collect(f::AbstractFunctional) = row(f)

function row(f::AbstractFunctional)
    result = Array(eltype(f), size(f))
    row!(result, f)
end

function row!(result, f::AbstractFunctional)
    @assert length(result) == length(f)

    set = src(f)
    coef = zeros(set)
    for (i,j) in enumerate(eachindex(set))
        coef[j] = 1
        result[i] = apply(f, coef)
        coef[j] = 0
    end
    result
end


"""An EvaluationFunctional represents a point evaluation."""
immutable EvaluationFunctional{N,T,SRC,ELT} <: AbstractFunctional{ELT}
    src ::  SRC
    x   ::  Vec{N,T}

    EvaluationFunctional(src::FunctionSet{N,T}, x::Vec{N,T}) = new(src, x)
end

EvaluationFunctional{N,T}(src::FunctionSet{N,T}, x::Vec{N,T}) =
    EvaluationFunctional{N,T,typeof(src),eltype(src)}(src, x)

EvaluationFunctional{T,S <: Number}(src::FunctionSet{1,T}, x::S) = EvaluationFunctional(src, Vec{1,T}(x))

apply(f::EvaluationFunctional, coef) = call_expansion(src(f), coef, f.x)




"A CompositeFunctional represents a functional times an operator."
immutable CompositeFunctional{ELT} <: AbstractFunctional{ELT}
    functional  ::  AbstractFunctional
    operator    ::  AbstractOperator

    # Scratch space to hold the result of the operator
    scratch

    function CompositeFunctional(functional, operator)
        scratch = zeros(dest(op))
        new(functional, operator, scratch)
    end
end

function CompositeFunctional(functional::AbstractFunctional, operator::AbstractOperator)
    ELT = promote_type(eltype(functional), eltype(operator))
    CompositeFunctional{ELT}(functional, operator)
end

functional(f::CompositeFunctional) = f.functional

operator(f::CompositeFunctional) = f.operator

src(f::CompositeFunctional) = src(operator(f))

apply(f::CompositeFunctional, coef) =
    apply_composite_functional(f.functional, f.operator, f.scratch, coef)

function apply_composite_functional(functional, operator, scratch, coef)
    apply!(operator, scratch, coef)
    apply(functional, scratch)
end

(*)(f::AbstractFunctional, op::AbstractOperator) = CompositeFunctional(f, op)

(*)(f::CompositeFunctional, op::AbstractOperator) = CompositeFunctional(functional(f), operator(f) * op)
