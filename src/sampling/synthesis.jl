# synthesis.jl

"""
The synthesis operator maps a discrete set of coefficients to a function
in the span of a dictionary.
"""
struct SynthesisOperator <: GenericOperator
    dictionary  ::  Dictionary
end

src(op::SynthesisOperator) = DiscreteSet

dictionary(op::SynthesisOperator) = op.dictionary

apply(op::SynthesisOperator, coef) = Expansion(dictionary(op), coef)

(*)(op::SynthesisOperator, coef) = apply(op, coef)
