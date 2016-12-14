# derived_set.jl

"""
A DerivedSet is a set that derives from an underlying set. The abstract type
derived sets implements a lot of the interface of a function set by delegating
to the underlying set.
"""
abstract DerivedSet{N,T} <: FunctionSet{N,T}

###########################################################################
# Warning: derived sets implements all functionality by delegating to the
# underlying set, as if the derived set does not want to change any of that
# behaviour. This may result in incorrect defaults. The concrete set should
# override any functionality that is changed by it.
###########################################################################

# Assume the concrete set has a field called set -- override if it doesn't
superset(s::DerivedSet) = s.superset

# The concrete subset should implement similar_set, as follows:
#
# similar_set(s::ConcreteDerivedSet, s2::FunctionSet) = ConcreteDerivedSet(s2)
#
# This function calls the constructor of the concrete set. We can then
# generically implement other methods that would otherwise call a constructor,
# such as resize and promote_eltype.

resize(s::DerivedSet, n) = similar_set(s, resize(superset(s),n))

set_promote_eltype{N,T,S}(s::DerivedSet{N,T}, ::Type{S}) =
    similar_set(s, promote_eltype(superset(s), S))

# Delegation of properties
for op in (:isreal, :is_basis, :is_frame, :is_orthogonal, :is_biorthogonal, :is_discrete)
    @eval $op(s::DerivedSet) = $op(superset(s))
end

# Delegation of feature methods
for op in (:has_derivative, :has_antiderivative, :has_grid, :has_extension)
    @eval $op(s::DerivedSet) = $op(superset(s))
end
# has_transform has extra arguments
has_grid_transform(s::DerivedSet, dgs, grid) = has_grid_transform(superset(s), dgs, grid)

# When getting started with a discrete set, you may want to write:
# has_derivative(s::ConcreteSet) = false
# has_antiderivative(s::ConcreteSet) = false
# has_grid(s::ConcreteSet) = false
# has_transform(s::ConcreteSet) = false
# has_transform(s::ConcreteSet, dgs) = false
# has_extension(s::ConcreteSet) = false
# ... and then implement those operations one by one and remove the definitions.

zeros(ELT::Type, s::DerivedSet) = zeros(ELT, superset(s))


# Delegation of methods
for op in (:length, :extension_size, :grid)
    @eval $op(s::DerivedSet) = $op(superset(s))
end

# Delegation of methods with an index parameter
for op in (:size,)
    @eval $op(s::DerivedSet, i) = $op(superset(s), i)
end

approx_length(s::DerivedSet, n) = approx_length(superset(s), n)

apply_map(s::DerivedSet, map) = similar_set(s, apply_map(superset(s), map))

#########################
# Indexing and iteration
#########################

native_index(s::DerivedSet, idx::Int) = native_index(superset(s), idx)

linear_index(s::DerivedSet, idxn) = linear_index(superset(s), idxn)

eachindex(s::DerivedSet) = eachindex(superset(s))

linearize_coefficients!(s::DerivedSet, coef_linear, coef_native) =
    linearize_coefficients!(superset(s), coef_linear, coef_native)

delinearize_coefficients!(s::DerivedSet, coef_native, coef_linear) =
    delinearize_coefficients!(superset(s), coef_native, coef_linear)

approximate_native_size(s::DerivedSet, size_l) = approximate_native_size(superset(s), size_l)

linear_size(s::DerivedSet, size_n) = linear_size(superset(s), size_n)

for op in (:left, :right)
    @eval $op{T}(s::DerivedSet{1,T}) = $op(superset(s))
    @eval $op{T}(s::DerivedSet{1,T}, idx) = $op(superset(s), idx)
end

eval_element(s::DerivedSet, idx, x) = eval_element(superset(s), idx, x)


#########################
# Wrapping of operators
#########################

for op in (:transform_set,)
    @eval $op(s::DerivedSet; options...) = $op(superset(s); options...)
end

for op in (:derivative_set,:antiderivative_set)
    @eval $op(s::DerivedSet, order; options...) = $op(superset(s), order; options...)
end


for op in (:extension_operator, :restriction_operator)
    @eval $op(s1::DerivedSet, s2::DerivedSet; options...) =
        wrap_operator(s1, s2, $op(superset(s1), superset(s2); options...))
end

# By default we return the underlying set when simplifying transforms
simplify_transform_pair(s::DerivedSet, grid::AbstractGrid) = (superset(s),grid)

# Simplify invocations of transform_from/to_grid with DerivedSet's
for op in ( (:transform_from_grid, :s1, :s2),
            (:transform_from_grid_pre, :s1, :s1),
            (:transform_from_grid_post, :s1, :s2))

    @eval function $(op[1])(s1, s2::DerivedSet, grid; options...)
        simple_s1, simple_s2, simple_grid = simplify_transform_sets(s1, s2, grid)
        operator = $(op[1])(simple_s1, simple_s2, simple_grid; options...)
        wrap_operator($(op[2]), $(op[3]), operator)
    end
end

for op in ( (:transform_to_grid, :s1, :s2),
            (:transform_to_grid_pre, :s1, :s1),
            (:transform_to_grid_post, :s1, :s2))

    @eval function $(op[1])(s1::DerivedSet, s2, grid; options...)
        simple_s1, simple_s2, simple_grid = simplify_transform_sets(s1, s2, grid)
        operator = $(op[1])(simple_s1, simple_s2, simple_grid; options...)
        wrap_operator($(op[2]), $(op[3]), operator)
    end
end


for op in (:differentiation_operator, :antidifferentiation_operator)
    @eval $op(s1::DerivedSet, s2::FunctionSet, order; options...) =
        wrap_operator(s1, s2, $op(superset(s1), s2, order; options...))
end


#########################
# Concrete set
#########################

"""
For testing purposes we define a concrete subset of DerivedSet. This set should
pass all interface tests and be functionally equivalent to the underlying set.
"""
immutable ConcreteDerivedSet{N,T} <: DerivedSet{N,T}
    superset ::  FunctionSet{N,T}
end

# Implementing similar_set is all it takes.

similar_set(s::ConcreteDerivedSet, s2::FunctionSet) = ConcreteDerivedSet(s2)
