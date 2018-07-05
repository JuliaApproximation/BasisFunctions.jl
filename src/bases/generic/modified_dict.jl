
"""
A `ModifiedDictionary` contains a dictionary and a modifier.

This type implements the full interface of a dictionary in a special way. The
method `foo` is delegated to the new method `mod_foo` with two extra arguments:
the dictionary and the modifier.

By default, `mod_foo` simply calles `foo` on the original dictionary and passes
on any argument. The modified dictionary thus inherits the behaviour of the
underlying dictionary. However, by using dispatch on the type of the modifier,
`mod_foo` may also choose to implement any other behaviour.
"""
abstract type ModifiedDictionary{S,T} <: Dictionary{S,T}
end

const ModifiedDict = ModifiedDictionary

struct TypedModifiedDictionary{D<:Dictionary,M,S,T} <: ModifiedDictionary{S,T}
    dict    ::  D
    modifier::  M
end

struct UntypedModifiedDictionary{S,T} <: ModifiedDictionary{S,T}
    dict    ::  Dictionary
    modifier
end

superdict(d::ModifiedDict) = d.dict

modifier(d::ModifiedDict) = d.modifier

basedict(d::ModifiedDict) = basedict(superdict(d))
basedict(d::Dictionary) = d

similar_dictionary(d::TypedModifiedDict, dict::Dictionary) = TypedModifiedDict(dict, modifier(d))

similar_dictionary(d::UntypedModifiedDict, dict::Dictionary) = UntypedModifiedDict(dict, modifier(d))


struct NoModification
end

mod_symbol(op::Symbol) = Symbol("mod_" * String(op))

size_modifier(d::ModifiedDict)        = size_modifier(modifier(d))
property_modifier(d::ModifiedDict)    = property_modifier(modifier(d))
derivative_modifier(d::ModifiedDict)  = derivative_modifier(modifier(d))
evaluation_modifier(d::ModifiedDict)  = evaluation_modifier(modifier(d))
coefficient_modifier(d::ModifiedDict) = coefficient_modifier(modifier(d))
composition_modifier(d::ModifiedDict) = composition_modifier(modifier(d))

for op in (:has_derivative, :has_antiderative, :unsafe_eval_element_derivative)
    mod_op = mod_symbol(op)
    @eval $op(d::ModifiedDict, args...) = $mod_op(derivative_modifier(d), superdict(d), args...)
    @eval $mod_op(mod::NoModification, superdict, args...) = $op(superdict, args...)
end

for op in (:length, :size, :resize, :extension_size, :ordering, :approx_length,
        :linear_index, :native_index, :eachindex, :approximate_native_size, :linear_size,
        :has_extension)
    mod_op = mod_symbol(op)
    @eval $op(d::ModifiedDict, args...) = $mod_op(size_modifier(d), superdict(d), args...)
    @eval $mod_op(mod::NoModification, superdict, args...) = $op(superdict, args...)
end

for op in (:coefficient_type, :linearize_coefficients!, :delinearize_coefficients!,
        :zeros)
    mod_op = mod_symbol(op)
    @eval $op(d::ModifiedDict, args...) = $mod_op(coefficient_modifier(d), superdict(d), args...)
    @eval $mod_op(mod::NoModification, superdict, args...) = $op(superdict, args...)
end

for op in (:has_grid, :has_grid_transform, :grid, :support, :in_support,
        :unsafe_eval_element, :unsafe_eval_element1)
    mod_op = mod_symbol(op)
    @eval $op(d::ModifiedDict, args...) = $mod_op(evaluation_modifier(d), superdict(d), args...)
    @eval $mod_op(mod::NoModification, superdict, args...) = $op(superdict, args...)
end

for op in (:is_composite, :numelements, :elements, :tail)
    mod_op = mod_symbol(op)
    @eval $op(d::ModifiedDict, args...) = $mod_op(composition_modifier(d), superdict(d), args...)
    @eval $mod_op(mod::NoModification, superdict, args...) = $op(superdict, args...)
end

for op in (:isreal, :is_basis, :is_frame, :is_orthogonal, :is_biorthogonal, :is_discrete)
    mod_op = mod_symbol(op)
    @eval $op(d::ModifiedDict, args...) = $mod_op(property_modifier(d), superdict(d), args...)
    @eval $mod_op(mod::NoModification, superdict, args...) = $op(superdict, args...)
end



#########################
# Wrapping of operators
#########################

for op in (:transform_dict,)
    mod_op = Symbol("mod_" * String(op))

    @eval $op(d::ModifiedDict, args...; options...) = $mod_op(d, superdict(d), modifier(d), args...; options...)
    @eval $mod_op(md, dict, mod, args...; options...) = $op(dict, args...; options...)
end

for op in (:derivative_dict, :antiderivative_dict)
    mod_op = Symbol("mod_" * String(op))

    @eval $op(d::ModifiedDict, args...; options...) = $mod_op(d, superdict(d), modifier(d), args...; options...)
    @eval $mod_op(md, dict, mod, args...; options...) =
        similar_dictionary(md, $op(dict, args...; options...))
end


for op in (:extension_operator, :restriction_operator)
    @eval $op(s1::DerivedDict, s2::DerivedDict; options...) =
        wrap_operator(s1, s2, $op(superdict(s1), superdict(s2); options...))
end

# By default we return the underlying set when simplifying transforms
simplify_transform_pair(d::ModifiedDict, grid::AbstractGrid) = (superdict(d),grid)

# Simplify invocations of transform_from/to_grid with DerivedDict's
for op in ( (:transform_from_grid, :s1, :s2),
            (:transform_from_grid_pre, :s1, :s1),
            (:transform_from_grid_post, :s1, :s2))

    @eval function $(op[1])(s1, s2::DerivedDict, grid; options...)
        simple_s1, simple_s2, simple_grid = simplify_transform_sets(s1, s2, grid)
        operator = $(op[1])(simple_s1, simple_s2, simple_grid; options...)
        wrap_operator($(op[2]), $(op[3]), operator)
    end
end

for op in ( (:transform_to_grid, :s1, :s2),
            (:transform_to_grid_pre, :s1, :s1),
            (:transform_to_grid_post, :s1, :s2))

    @eval function $(op[1])(s1::DerivedDict, s2, grid; options...)
        simple_s1, simple_s2, simple_grid = simplify_transform_sets(s1, s2, grid)
        operator = $(op[1])(simple_s1, simple_s2, simple_grid; options...)
        wrap_operator($(op[2]), $(op[3]), operator)
    end
end


for op in (:differentiation_operator, :antidifferentiation_operator)
    @eval $op(s1::DerivedDict, s2::DerivedDict, order; options...) =
        wrap_operator(s1, s2, $op(superdict(s1), superdict(s2), order; options...))
end

grid_evaluation_operator(set::DerivedDict, dgs::GridBasis, grid::AbstractGrid; options...) =
    wrap_operator(set, dgs, grid_evaluation_operator(superdict(set), dgs, grid; options...))

grid_evaluation_operator(set::DerivedDict, dgs::GridBasis, grid::AbstractSubGrid; options...) =
    wrap_operator(set, dgs, grid_evaluation_operator(superdict(set), dgs, grid; options...))

dot(d::ModifiedDict, f1::Function, f2::Function, nodes::Array=native_nodes(superdict(d)); options...) =
    dot(superdict(d), f1, f2, nodes; options...)



function stencil(d::ModifiedDict,S)
    A = Any[]
    push!(A,S[d])
    push!(A,"(")
    push!(A,superdict(d))
    push!(A,")")
    return recurse_stencil(d,A,S)
end
has_stencil(d::ModifiedDict) = true
