# concepts.jl

"""
An `OperatorConcept` represents an operator with a well-defined conceptual
meaning, but with a realization that depends on context.

The operator can be specialized to a certain type of arguments by invoking
`specialize`. For example, the `Differentiation` operator specializes to
a differentiation operator when applied to a dictionary.
"""
abstract type OperatorConcept <: GenericOperator
end


"""
The `Differentiation` operator represents differentiation of a certain `order`
with respect to a `variable`, determined by its index.
"""
struct Differentiation <: OperatorConcept
    # TODO: generalize differentiation_operator so that it accepts a variable
    variable    ::  Int
    order       ::  Int

    Differentiation(variable = 1, order = 1) = new(variable, order)
end

specialize(op::Differentiation, span::Span) =
    differentiation_operator(span, order = op.order)

specialize(op::Differentiation, dict::Dictionary) =
    differentiation_operator(Span(dict), order = op.order)

(*)(op::Differentiation, span::Span) = specialize(op, span)

apply(op::Differentiation, f::Expansion) =
    apply(specialize(op, Span(f)), f)
