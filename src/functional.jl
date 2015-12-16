# functional.jl


"""
AbstractFunctional is the supertype of all functionals in BasisFunctions.
Any functional has a source and an element type.

The dimensions of a functional are like a row: (1,length(src)).
"""
abstract AbstractFunctional{SRC,ELT}

src(f::AbstractFunctional) = f.src

eltype{SRC,ELT}(::Type{AbstractFunctional{SRC,ELT}}) = ELT
eltype{F <: AbstractFunctional}(::Type{F}) = eltype(super(F))

length(f::AbstractFunctional) = length(src(f))

size(f::AbstractFunctional) = (1,length(f))


apply(f::AbstractFunctional, coef) = apply(f, src(f), coef)

(*)(f::AbstractFunctional, coef::AbstractArray) = apply(f, coef)


collect(f::AbstractFunctional) = row(f)

function row(f::AbstractFunctional)
    result = Array(eltype(f), size(f))
    row!(result, f)
end

function row!(result, f::AbstractFunctional)
    n = length(src(f))
    
    @assert size(result) == (1,n)
    
    T = eltype(f)
    r = zeros(T,n)
    r_src = reshape(r, size(src(f)))
    for i = 1:n
        if (i > 1)
            r[i-1] = 0
        end
        r[i] = 1
        result[i] = apply(f, r_src)
    end
    result
end


"""An EvaluationFunctional represents a point evaluation."""
immutable EvaluationFunctional{N,T,SRC,ELT} <: AbstractFunctional{SRC,ELT}
    src ::  SRC
    x   ::  Vec{N,T}

    EvaluationFunctional(src::FunctionSet{N,T}, x::Vec{N,T}) = new(src, x)
end

EvaluationFunctional{N,T}(src::FunctionSet{N,T}, x::Vec{N,T}) =
    EvaluationFunctional{N,T,typeof(src),eltype(src)}(src, x)

EvaluationFunctional{T,S <: Number}(src::FunctionSet{1,T}, x::S) = EvaluationFunctional(src, Vec{1,T}(x))


apply(f::EvaluationFunctional, src, coef) = call_expansion(src, coef, f.x)




"A CompositeFunctional represents a functional times an operator."
immutable CompositeFunctional{F,OP,ID,SRC,ELT} <: AbstractFunctional{SRC,ELT}
    f   ::  F
    op  ::  OP

    scratch ::  Array{ELT,1}

    function CompositeFunctional(f::AbstractFunctional, op::AbstractOperator{SRC})
        scratch = Array(ELT, size(dest(op)))
        new(f, op, scratch)
    end
end

CompositeFunctional{SRC}(f::AbstractFunctional, op::AbstractOperator{SRC}) =
    CompositeFunctional{typeof(f), typeof(op), index_dim(dest(op)), SRC, eltype(op)}(f, op)

functional(f::CompositeFunctional) = f.f

operator(f::CompositeFunctional) = f.op

src(f::CompositeFunctional) = src(operator(f))


(*)(f::AbstractFunctional, op::AbstractOperator) = CompositeFunctional(f, op)

(*)(f::CompositeFunctional, op::AbstractOperator) = CompositeFunctional(functional(f), operator(f) * op)


function apply(f::CompositeFunctional, src, coef)
    apply!(operator(f), f.scratch, coef)
    apply(functional(f), f.scratch)
end



