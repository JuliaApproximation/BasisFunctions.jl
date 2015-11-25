# functional.jl


"""
AbstractFunctional is the supertype of all functionals in BasisFunctions.
Any functional has a source and an element type.

The dimensions of a functional are like a row: (1,length(src)).
"""
abstract AbstractFunctional{SRC,ELT}

src(f::AbstractFunctional) = f.src

eltype{SRC,ELT}(f::AbstractFunctional{SRC,ELT}) = ELT
eltype{SRC,ELT}(::Type{AbstractFunctional{SRC,ELT}}) = ELT
eltype{F <: AbstractFunctional}(::Type{F}) = eltype(super(F))


"""An EvaluationFunctional represents a point evaluation."""
immutable EvaluationFunctional{N,T,SRC,ELT} <: AbstractFunctional{SRC,ELT}
    src ::  SRC
    x   ::  Vec{N,T}

    EvaluationFunctional(src::FunctionSet{N,T}, x::Vec{N,T}) = new(src, x)
end

EvaluationFunctional{N,T}(src::FunctionSet{N,T}, x::Vec{N,T}) =
    EvaluationFunctional{N,T,typeof(src),eltype(src)}(src, x)


# TODO: use generated function to get rid of splatting
apply(f::EvaluationFunctional, coef) = call(f.src, coef, f.x...)


