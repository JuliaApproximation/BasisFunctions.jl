# modified_dict.jl

"""
A modified dictionary contains a dictionary and a modifier.

The interface methods of a dictionary are delegated to the underlying dictionary,
with the modifier passed as extra argument. This may result in a modification
of the behaviour of the original dictionary.
"""
abstract type ModifiedDictionary{S,T} <: Dictionary{S,T}
end

const ModifiedSpan{A,S,T,D <: ModifiedDictionary} = Span{A,S,T,D}

abstract type DictModifier
end


"A `TypedModifiedDictionary` has a fully typed dictionary and modifier."
struct TypedModifiedDictionary{D,M,S,T} <: ModifiedDictionary{S,T}
    dictionary  ::  D
    modifier    ::  M
end

TypedModifiedDictionary(dict::Dictionary{S,T}, modifier) where {S,T} =
    TypedModifiedDictionary{typeof(dict),typeof(modifier),S,T}(dict, modifier)

struct UntypedModifiedDictionary{S,T} <: ModifiedDictionary{S,T}
    dictionary  ::  Dictionary
    modifier
end


superdict(dict::ModifiedDictionary) = dict.dictionary

modifier(dict::ModifiedDictionary) = dict.modifier

coefficient_type(d::ModifiedDictionary) = mod_coefficient_type(modifier(d), superdict(d))
mod_coefficient_type(mod, d) = coefficient_type(d)

# Delegation of properties
for (op,modop) in ( (:isreal, :mod_isreal), (:is_basis, :mod_isbasis),
        (:is_frame, :mod_isframe), (:is_orthogonal, :mod_is_orthogonal),
        (:is_biorthogonal, :mod_is_biorthogonal), (:is_discrete, :mod_isdiscrete))
    @eval ($op)(dict::ModifiedDictionary) = ($modop)(modifier(dict), superdict(dict))
    @eval $(modop)(mod, dict) = ($op)(dict)
end

# Delegation of feature methods
for (op, modop) in ((:has_derivative, :mod_has_derivative),
        (:has_antiderivative, :mod_has_antiderivative),
        (:has_grid, :mod_has_grid), (:has_extension, :mod_has_extension))
    @eval $op(dict::ModifiedDictionary) = $modop(modifier(dict), superdict(dict))
    @eval $modop(mod, dict) = ($op)(dict)
end

# has_transform has extra arguments
has_grid_transform(dict::ModifiedDictionary, gs, grid) =
    mod_has_grid_transform(modifier(dict), superdict(dict), gs, grid)
mod_has_grid_transform(mod, dict, gs, grid) = has_grid_transform(dict, gs, grid)

zeros(::Type{T}, dict::ModifiedDictionary) where {T} =
    mod_zeros(T, modifier(dict), superdict(dict))
mod_zeros(mod, dict) = zeros(T, dict)

# Delegation of methods
for (op, modop) in ((:length, :mod_length),
        (:extension_size, :mod_extension_size), (:size, :mod_size),
        (:grid, :mod_grid), (:is_composite, :mod_is_composite),
        (:nb_elements, :mod_nb_elements), (:elements, :mod_elements),
        (:tail, :mod_tail), (:ordering, :mod_ordering))
    @eval $op(dict::ModifiedDictionary) = $modop(modifier(dict), superdict(dict))
    @eval $modop(mod, dict) = $op(dict)
end

# Delegation of methods with an index parameter
for (op, modop) in ((:size, :mod_size), (:element, :mod_element))
    @eval $op(dict::ModifiedDictionary, idx) =
        $modop(modifier(dict), superdict(dict), idx)
    @eval $modop(mod, dict, idx) = $op(dict, idx)
end

approx_length(dict::ModifiedDictionary, n::Int) =
    mod_approx_length(modifier(dict), superdict(dict), n)
mod_approx_length(mod, dict, n) = approx_length(dict, n)

apply_map(dict::ModifiedDictionary, map) =
    similar_dictionary(dict, apply_map(superdict(dict), map))

in_support(dict::ModifiedDictionary, i, x) =
    mod_in_support(modifier(dict), superdict(dict), i, x)
mod_in_support(mod, dict, i, x) = in_support(dict, i, x)


#########################
# Indexing and iteration
#########################

native_index(dict::ModifiedDictionary, idx) = mod_native_index(modifier(dict), superdict(dict), idx)
mod_native_index(mod, dict, idx) = native_index(dict, idx)

linear_index(dict::ModifiedDictionary, idxn) = mod_linear_index(modifier(dict), superdict(dict), idxn)
mod_linear_index(mod, dict, idxn) = linear_index(dict, idxn)

eachindex(dict::ModifiedDictionary) = mod_eachindex(modifier(dict), superdict(dict))
mod_eachindex(mod, dict) = eachindex(dict)

linearize_coefficients!(dict::ModifiedDictionary, coef_linear::Vector, coef_native) =
    linearize_coefficients!(superdict(dict), coef_linear, coef_native)

delinearize_coefficients!(dict::ModifiedDictionary, coef_native, coef_linear::Vector) =
    delinearize_coefficients!(superdict(dict), coef_native, coef_linear)

approximate_native_size(dict::ModifiedDictionary, size_l) =
    mod_approximate_native_size(modifier(dict), superdict(dict), size_l)
mod_approximate_native_size(mod, dict, size_l) = approximate_native_size(dict, size_l)

linear_size(dict::ModifiedDictionary, size_n) =
    mod_linear_size(modifier(dict), superdict(dict), size_n)
mod_linear_size(mod, dict, size_n) = linear_size(dict, size_n)

for op in (:left, :right)
    @eval $op(dict::ModifiedDictionary) = $op(superdict(dict))
    @eval $op(dict::ModifiedDictionary, idx) = $op(superdict(dict), idx)
end

unsafe_eval_element(dict::ModifiedDictionary, idx, x) =
    mod_unsafe_eval_element(modifier(dict), superdict(dict), idx, x)
mod_unsafe_eval_element(mod, dict, idx, x) = unsafe_eval_element(dict, idx, x)

unsafe_eval_element_derivative(dict::ModifiedDictionary, idx, x) =
    mod_unsafe_eval_element_derivative(modifier(dict), superdict(dict), idx, x)
mod_unsafe_eval_element_derivative(mod, dict, idx, x) =
    unsafe_eval_element_derivative(dict, idx, x)


#########################
# Wrapping of operators
#########################

for op in (:transform_space,)
    @eval $op(s::ModifiedSpan; options...) = $op(superspan(s); options...)
end

for op in (:derivative_space, :antiderivative_space)
    @eval $op(s::ModifiedSpan, order; options...) = similar_span(s, $op(superspan(s), order; options...))
end


for op in (:extension_operator, :restriction_operator)
    @eval $op(s1::ModifiedSpan, s2::ModifiedSpan; options...) =
        wrap_operator(s1, s2, $op(superspan(s1), superspan(s2); options...))
end

# By default we return the underlying set when simplifying transforms
simplify_transform_pair(s::ModifiedDictionary, grid::AbstractGrid) = (superdict(s),grid)

# Simplify invocations of transform_from/to_grid with ModifiedDictionary's
for op in ( (:transform_from_grid, :s1, :s2),
            (:transform_from_grid_pre, :s1, :s1),
            (:transform_from_grid_post, :s1, :s2))

    @eval function $(op[1])(s1, s2::ModifiedSpan, grid; options...)
        simple_s1, simple_s2, simple_grid = simplify_transform_spaces(s1, s2, grid)
        operator = $(op[1])(simple_s1, simple_s2, simple_grid; options...)
        wrap_operator($(op[2]), $(op[3]), operator)
    end
end

for op in ( (:transform_to_grid, :s1, :s2),
            (:transform_to_grid_pre, :s1, :s1),
            (:transform_to_grid_post, :s1, :s2))

    @eval function $(op[1])(s1::ModifiedSpan, s2, grid; options...)
        simple_s1, simple_s2, simple_grid = simplify_transform_spaces(s1, s2, grid)
        operator = $(op[1])(simple_s1, simple_s2, simple_grid; options...)
        wrap_operator($(op[2]), $(op[3]), operator)
    end
end


for op in (:differentiation_operator, :antidifferentiation_operator)
    @eval $op(s1::ModifiedSpan, s2::ModifiedSpan, order; options...) =
        wrap_operator(s1, s2, $op(superspan(s1), superspan(s2), order; options...))
end

grid_evaluation_operator(set::ModifiedSpan, dgs::DiscreteGridSpace, grid::AbstractGrid; options...) =
    wrap_operator(set, dgs, grid_evaluation_operator(superspan(set), dgs, grid; options...))

grid_evaluation_operator(set::ModifiedSpan, dgs::DiscreteGridSpace, grid::AbstractSubGrid; options...) =
    wrap_operator(set, dgs, grid_evaluation_operator(superspan(set), dgs, grid; options...))

dot(s::ModifiedSpan, f1::Function, f2::Function, nodes::Array=native_nodes(superdict(s)); options...) =
    dot(superspan(s), f1, f2, nodes; options...)


# For testing purposes we define a modifer that does nothing.
struct NoModifier <: DictModifier
end
