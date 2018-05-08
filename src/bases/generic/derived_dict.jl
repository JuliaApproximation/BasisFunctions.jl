# derived_set.jl

"""
A `DerivedDict` is a dictionary that inherits most of its behaviour from
an underlying dictionary.

The abstract type for derived dictionaries implements the interface of a
dictionary using composition and delegation. Concrete derived dictionaries
may override functions to specialize behaviour. For example, a mapped dictionary
may override the evaluation routine to apply the map first.
"""
abstract type DerivedDict{S,T} <: Dictionary{S,T}
end

const DerivedSpan{A,S,T,D <: DerivedDict} = Span{A,S,T,D}

###########################################################################
# Warning: derived sets implements all functionality by delegating to the
# underlying set, as if the derived set does not want to change any of that
# behaviour. This may result in incorrect defaults. The concrete set should
# override any functionality that is changed by it.
###########################################################################

# Assume the concrete set has a field called set -- override if it doesn't
superdict(s::DerivedDict) = s.superdict

superdict(s::DerivedSpan) = superdict(dictionary(s))

"Return the span of the superdict of the given derived set."
superspan(s::DerivedSpan) = Span(superdict(s), coeftype(s))

# The concrete subset should implement similar_dictionary, as follows:
#
# similar_dictionary(s::ConcreteDerivedDict, s2::Dictionary) = ConcreteDerivedDict(s2)
#
# This function calls the constructor of the concrete set. We can then
# generically implement other methods that would otherwise call a constructor,
# such as resize and promote_eltype.

similar_span(s::DerivedSpan, s2::Span) = Span(similar_dictionary(dictionary(s), dictionary(s2)), coeftype(s2))

resize(s::DerivedDict, n) = similar_dictionary(s, resize(superdict(s),n))

# To avoid ambiguity with a similar definition for abstract type Dictionary:
resize(s::DerivedDict, n::Tuple{Int}) = resize(s, n[1])

dict_promote_domaintype(s::DerivedDict{T}, ::Type{S}) where {T,S} =
    similar_dictionary(s, promote_domaintype(superdict(s), S))

for op in (:coefficient_type,)
    @eval $op(s::DerivedDict) = $op(superdict(s))
end

# Delegation of properties
for op in (:isreal, :is_basis, :is_frame, :is_orthogonal, :is_biorthogonal, :is_discrete)
    @eval $op(s::DerivedDict) = $op(superdict(s))
end

# Delegation of feature methods
for op in (:has_derivative, :has_antiderivative, :has_grid, :has_extension)
    @eval $op(s::DerivedDict) = $op(superdict(s))
end
# has_transform has extra arguments
has_grid_transform(s::DerivedDict, gs, grid) = has_grid_transform(superdict(s), gs, grid)

# When getting started with a discrete set, you may want to write:
# has_derivative(s::ConcreteSet) = false
# has_antiderivative(s::ConcreteSet) = false
# has_grid(s::ConcreteSet) = false
# has_transform(s::ConcreteSet) = false
# has_transform(s::ConcreteSet, dgs::GridBasis) = false
# has_extension(s::ConcreteSet) = false
# ... and then implement those operations one by one and remove the definitions.

zeros(::Type{T}, s::DerivedDict) where {T} = zeros(T, superdict(s))


# Delegation of methods
for op in (:length, :extension_size, :size, :grid, :is_composite, :nb_elements,
    :elements, :tail, :ordering, :support)
    @eval $op(s::DerivedDict) = $op(superdict(s))
end

# Delegation of methods with an index parameter
for op in (:size, :element, :support)
    @eval $op(s::DerivedDict, i) = $op(superdict(s), i)
end

approx_length(s::DerivedDict, n::Int) = approx_length(superdict(s), n)

apply_map(s::DerivedDict, map) = similar_dictionary(s, apply_map(superdict(s), map))

in_support(set::DerivedDict, i, x) = in_support(superdict(set), i, x)

# To avoid an ambiguity with a similar definition for abstract type Dictionary:
in_support(set::DerivedDict, idx, x::T) where {T <: Complex} =
    imag(x) == 0 && in_support(superdict(set), idx, real(x))

#########################
# Indexing and iteration
#########################

native_index(dict::DerivedDict, idx) = native_index(superdict(dict), idx)

linear_index(dict::DerivedDict, idxn) = linear_index(superdict(dict), idxn)

eachindex(dict::DerivedDict) = eachindex(superdict(dict))

linearize_coefficients!(s::DerivedDict, coef_linear::Vector, coef_native) =
    linearize_coefficients!(superdict(s), coef_linear, coef_native)

delinearize_coefficients!(s::DerivedDict, coef_native, coef_linear::Vector) =
    delinearize_coefficients!(superdict(s), coef_native, coef_linear)

approximate_native_size(s::DerivedDict, size_l) = approximate_native_size(superdict(s), size_l)

linear_size(s::DerivedDict, size_n) = linear_size(superdict(s), size_n)


unsafe_eval_element(s::DerivedDict, idx, x) = unsafe_eval_element(superdict(s), idx, x)

unsafe_eval_element_derivative(s::DerivedDict, idx, x) =
    unsafe_eval_element_derivative(superdict(s), idx, x)


#########################
# Wrapping of operators
#########################

for op in (:transform_space,)
    @eval $op(s::DerivedSpan; options...) = $op(superspan(s); options...)
end

for op in (:derivative_space, :antiderivative_space)
    @eval $op(s::DerivedSpan, order; options...) = similar_span(s, $op(superspan(s), order; options...))
end


for op in (:extension_operator, :restriction_operator)
    @eval $op(s1::DerivedSpan, s2::DerivedSpan; options...) =
        wrap_operator(s1, s2, $op(superspan(s1), superspan(s2); options...))
end

# By default we return the underlying set when simplifying transforms
simplify_transform_pair(s::DerivedDict, grid::AbstractGrid) = (superdict(s),grid)

# Simplify invocations of transform_from/to_grid with DerivedDict's
for op in ( (:transform_from_grid, :s1, :s2),
            (:transform_from_grid_pre, :s1, :s1),
            (:transform_from_grid_post, :s1, :s2))

    @eval function $(op[1])(s1, s2::DerivedSpan, grid; options...)
        simple_s1, simple_s2, simple_grid = simplify_transform_spaces(s1, s2, grid)
        operator = $(op[1])(simple_s1, simple_s2, simple_grid; options...)
        wrap_operator($(op[2]), $(op[3]), operator)
    end
end

for op in ( (:transform_to_grid, :s1, :s2),
            (:transform_to_grid_pre, :s1, :s1),
            (:transform_to_grid_post, :s1, :s2))

    @eval function $(op[1])(s1::DerivedSpan, s2, grid; options...)
        simple_s1, simple_s2, simple_grid = simplify_transform_spaces(s1, s2, grid)
        operator = $(op[1])(simple_s1, simple_s2, simple_grid; options...)
        wrap_operator($(op[2]), $(op[3]), operator)
    end
end


for op in (:differentiation_operator, :antidifferentiation_operator)
    @eval $op(s1::DerivedSpan, s2::DerivedSpan, order; options...) =
        wrap_operator(s1, s2, $op(superspan(s1), superspan(s2), order; options...))
end

grid_evaluation_operator(set::DerivedSpan, dgs::DiscreteGridSpace, grid::AbstractGrid; options...) =
    wrap_operator(set, dgs, grid_evaluation_operator(superspan(set), dgs, grid; options...))

grid_evaluation_operator(set::DerivedSpan, dgs::DiscreteGridSpace, grid::AbstractSubGrid; options...) =
    wrap_operator(set, dgs, grid_evaluation_operator(superspan(set), dgs, grid; options...))

dot(s::DerivedSpan, f1::Function, f2::Function, nodes::Array=native_nodes(superdict(s)); options...) =
    dot(superspan(s), f1, f2, nodes; options...)


#########################
# Concrete dict
#########################

"""
For testing purposes we define a concrete subset of DerivedDict. This set should
pass all interface tests and be functionally equivalent to the underlying set.
"""
struct ConcreteDerivedDict{S,T} <: DerivedDict{S,T}
    superdict   ::  Dictionary{S,T}
end

# Implementing similar_dictionary is all it takes.

similar_dictionary(s::ConcreteDerivedDict, s2::Dictionary) = ConcreteDerivedDict(s2)

function stencil(d::DerivedDict,S)
    A = Any[]
    push!(A,S[d])
    push!(A,"(")
    push!(A,superdict(d))
    push!(A,")")
    return recurse_stencil(d,A,S)
end
has_stencil(d::DerivedDict) = true
