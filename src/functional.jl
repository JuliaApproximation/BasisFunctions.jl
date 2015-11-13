# functional.jl


"""
An AbstractFunctional is the supertype of all functionals in BasisFunctions.
Any functional has a source (SRC parameter) and a destination type (DT).
"""
abstract AbstractFunctional{SRC <: FunctionSet,DT <: Number}

immutable EvaluationFunctional
end
