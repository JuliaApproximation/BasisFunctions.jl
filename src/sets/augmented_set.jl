# augmented_set.jl

"""
An AugmentedSet represents some function f(x) times an existing set.
"""
immutable AugmentedSet{S,F,N,T} <: DerivedSet{N,T}
    set     ::  S
    fun     ::  F
end
# Perhaps we can remove the type parameters?
# It would hurt evaluation of an element, but it would not impact the cost of
# any of the operators.

AugmentedSet{N,T}(s::FunctionSet{N,T}, fun) = AugmentedSet{typeof(s),typeof(fun),N,T}(s, fun)

fun(s::AugmentedSet) = s.fun

similar_set(s::AugmentedSet, s2::FunctionSet) = AugmentedSet(s2, fun(s))

name(s::AugmentedSet) = _name(s, set(s), fun(s))
_name(s::AugmentedSet, set, fun::AbstractFunction) = "An augmented set based on " * name(set)
_name(s::AugmentedSet, set, fun::Function) = name(fun) * " * " * name(set)

isreal(s::AugmentedSet) = _isreal(s, set(s), fun(s))
_isreal(s::AugmentedSet, set, fun::AbstractFunction) = isreal(set) && isreal(fun)
_isreal(s::AugmentedSet, set, fun::Function) = isreal(set)

has_derivative(s::AugmentedSet) = has_derivative(set(s)) && has_derivative(fun(s))

# We can not compute antiderivatives in general.
has_antiderivative(s::AugmentedSet) = false

# Evaluating basis functions: we multiply by the function of the set
eval_element(set::AugmentedSet, idx, x) = set.fun(x) * eval_element(set.set, idx, x)
eval_element(set::AugmentedSet, idx, x::SVector) = set.fun(x...) * eval_element(set.set, idx, x)

# You can create an AugmentedSet by multiplying a function with a set, using
# left multiplication.
# We support any Julia function:
(*)(f::Function, s::FunctionSet) = AugmentedSet(s, f)
# and our own functors:
(*)(f::AbstractFunction, s::FunctionSet) = AugmentedSet(s, f)


function transform_post_operator(src::AugmentedSet, dest::DiscreteGridSpace; options...)
    f = fun(src)
    DiagonalOperator(dest, dest, [f(x) for x in grid(dest)])
end

transform_pre_operator(src::DiscreteGridSpace, dest::AugmentedSet; options...) =
	inv(transform_post_operator(dest, src; options...))


function derivative_set(src::AugmentedSet, order)
    @assert order == 1

    s = set(src)
    f = fun(src)
    f_prime = derivative(f)
    s_prime = derivative_set(s, order)
    (f_prime * s) ⊕ (f * s_prime)
end

# Assume order = 1...
function differentiation_operator(s1::AugmentedSet, s2::MultiSet, order; options...)
    @assert order == 1
    @assert s2 == derivative_set(s1, order)

    I = IdentityOperator(s1, element(s2, 1))
    D = differentiation_operator(set(s1))
    DW = wrap_operator(s1, element(s2, 2), D)
    block_column_operator([I,DW])
end
