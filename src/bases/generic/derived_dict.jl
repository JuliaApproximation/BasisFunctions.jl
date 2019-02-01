
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

###########################################################################
# Warning: derived sets implements all functionality by delegating to the
# underlying set, as if the derived set does not want to change any of that
# behaviour. This may result in incorrect defaults. The concrete set should
# override any functionality that is changed by it.
###########################################################################

# Assume the concrete set has a field called set -- override if it doesn't
superdict(s::DerivedDict) = s.superdict

# The concrete subset should implement similar_dictionary, as follows:
#
# similar_dictionary(s::ConcreteDerivedDict, s2::Dictionary) = ConcreteDerivedDict(s2)
#
# This function calls the constructor of the concrete set. We can then
# generically implement other methods that would otherwise call a constructor,
# such as resize and promote_eltype.

similar(d::DerivedDict, ::Type{T}, dims::Int...) where {T} =
    similar_dictionary(d, similar(superdict(d), T, dims))

promote_coefficienttype(dict::DerivedDict, ::Type{T}) where {T} =
    similar_dictionary(dict, promote_coefficienttype(superdict(dict), T))

promote_coefficienttype(dict::DerivedDict, ::Type{Any}) = dict

for op in (:coefficienttype,)
    @eval $op(s::DerivedDict) = $op(superdict(s))
end

# Delegation of properties
for op in (:isreal, :isbasis, :isframe, :isdiscrete)
    @eval $op(s::DerivedDict) = $op(superdict(s))
end

for op in (:isorthogonal, :isbiorthogonal, :isorthonormal)
    @eval $op(s::DerivedDict, m::Measure) = $op(superdict(s), m)
end



# Delegation of feature methods
for op in (:hasderivative, :hasantiderivative, :hasinterpolationgrid, :hasextension)
    @eval $op(s::DerivedDict) = $op(superdict(s))
end
# hastransform has extra arguments
hasgrid_transform(s::DerivedDict, gs, grid) = hasgrid_transform(superdict(s), gs, grid)

# When getting started with a discrete set, you may want to write:
# hasderivative(s::ConcreteSet) = false
# hasantiderivative(s::ConcreteSet) = false
# hasinterpolationgrid(s::ConcreteSet) = false
# hastransform(s::ConcreteSet) = false
# hastransform(s::ConcreteSet, dgs::GridBasis) = false
# hasextension(s::ConcreteSet) = false
# ... and then implement those operations one by one and remove the definitions.

zeros(::Type{T}, s::DerivedDict) where {T} = zeros(T, superdict(s))
tocoefficientformat(a, d::DerivedDict) = tocoefficientformat(a, superdict(d))

# Delegation of methods
for op in (:length, :extension_size, :size, :interpolation_grid, :iscomposite, :numelements,
    :elements, :tail, :ordering, :support)
    @eval $op(s::DerivedDict) = $op(superdict(s))
end

# Delegation of methods with an index parameter
for op in (:size, :element, :support)
    @eval $op(s::DerivedDict, i) = $op(superdict(s), i)
end

approx_length(s::DerivedDict, n::Int) = approx_length(superdict(s), n)

apply_map(s::DerivedDict, map) = similar_dictionary(s, apply_map(superdict(s), map))

dict_in_support(set::DerivedDict, i, x) = in_support(superdict(set), i, x)


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

for op in (:transform_dict,)
    @eval $op(s::DerivedDict; options...) = $op(superdict(s); options...)
end

for op in (:derivative_dict, :antiderivative_dict)
    @eval $op(s::DerivedDict, order; options...) = similar_dictionary(s, $op(superdict(s), order; options...))
end


for op in (:extension_operator, :restriction_operator)
    @eval $op(s1::DerivedDict, s2::DerivedDict; T = op_eltype(s1,s2), options...) =
        wrap_operator(s1, s2, $op(superdict(s1), superdict(s2); T = T, options...))
end

function transform_from_grid(s1::GridBasis, s2::DerivedDict, grid; T = op_eltype(s1,s2), options...)
    op = transform_from_grid(s1, superdict(s2), grid; T=T, options...)
    wrap_operator(s1, s2, op)
end

function transform_to_grid(s1::DerivedDict, s2::GridBasis, grid; T = op_eltype(s1,s2), options...)
    op = transform_to_grid(superdict(s1), s2, grid; T=T, options...)
    wrap_operator(s1, s2, op)
end


for op in (:differentiation_operator, :antidifferentiation_operator)
    @eval $op(s1::DerivedDict, s2::DerivedDict, order; T = op_eltype(s1,s2), options...) =
        wrap_operator(s1, s2, $op(superdict(s1), superdict(s2), order; T=T, options...))
end

pseudodifferential_operator(s::DerivedDict, symbol::Function; options...) = pseudodifferential_operator(s,s,symbol;options...)
pseudodifferential_operator(s1::DerivedDict,s2::DerivedDict, symbol::Function; options...) = wrap_operator(s1,s2,pseudodifferential_operator(superdict(s1),superdict(s2),symbol; options...))


function grid_evaluation_operator(dict::DerivedDict, gb::GridBasis, grid;
        T = op_eltype(dict, gb), options...)
    A = grid_evaluation_operator(superdict(dict), gb, grid; T=T, options...)
    wrap_operator(dict, gb, A)
end

## Printing

hasstencil(dict::DerivedDict) = true
stencilarray(dict::DerivedDict) = [modifiersymbol(dict), "(", superdict(dict), ")"]
modifiersymbol(dict::DerivedDict) = PrettyPrintSymbol{:DefDerive}(dict)
name(::PrettyPrintSymbol{:DefDerive}) = "Derived"

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
