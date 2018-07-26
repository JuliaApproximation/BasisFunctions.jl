# concepts.jl

"""
An `OperatorConcept` represents an operator with a well-defined conceptual
meaning, but with a realization that depends on context.
"""
abstract type OperatorConcept
end


struct Interpolation
    src     ::  Dictionary
    dest    ::  Dictionary
end
