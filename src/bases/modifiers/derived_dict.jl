
"""
A `DerivedDict` is a dictionary that inherits most of its behaviour from
an underlying dictionary.

The abstract type for derived dictionaries implements the interface of a
dictionary using composition and delegation. Concrete derived dictionaries
may override functions to specialize behaviour. For example, a mapped dictionary
may override the evaluation routine to apply the map first.
"""
abstract type DerivedDict{S,T} <: Dictionary{S,T} end

abstract type HighlySimilarDerivedDict{S,T} <: DerivedDict{S,T} end

###########################################################################
# Warning: derived sets implements all functionality by delegating to the
# underlying set, as if the derived set does not want to change any of that
# behaviour. This may result in incorrect defaults. The concrete set should
# override any functionality that is changed by it.
# Update: more functionality is copied for HighlySimilarDerivedDict, less
# for the derived dictionaries.
###########################################################################

# Assume the concrete set has a field called superdict -- override if it doesn't
superdict(d::DerivedDict) = d.superdict

# The concrete subset should implement similardictionary, as follows:
#
# similardictionary(d::ConcreteDerivedDict, d2::Dictionary) = ConcreteDerivedDict(d2)
#
# This function calls the constructor of the concrete set. We can then
# generically implement other methods that would otherwise call a constructor,
# such as resize and promote_eltype.

similar(d::DerivedDict, ::Type{T}, dims::Int...) where {T} =
    similardictionary(d, similar(superdict(d), T, dims))

resize(d::DerivedDict, dims) = similardictionary(d, resize(superdict(d), dims))

for op in (:coefficienttype,)
    @eval $op(d::DerivedDict) = $op(superdict(d))
end

# Delegation of properties
for op in (:isreal, :isbasis, :isframe, :isdiscrete, :measure)
    @eval $op(d::DerivedDict) = $op(superdict(d))
end

for op in (:isorthogonal, :isbiorthogonal, :isorthonormal)
    @eval $op(d::DerivedDict, μ::Measure) = $op(superdict(d), μ)
end



# Delegation of feature methods
for op in (:hasderivative, :hasantiderivative, :hasinterpolationgrid, :hasextension)
    @eval $op(Φ::DerivedDict) = $op(superdict(Φ))
end
for op in (:hasderivative, :hasantiderivative)
    @eval $op(Φ::DerivedDict, order) = $op(superdict(Φ), order)
end

# hastransform has extra arguments
hasgrid_transform(d::DerivedDict, gb, grid) = hasgrid_transform(superdict(d), gb, grid)

# When getting started with a discrete dictionary, you may want to write:
# hasderivative(d::ConcreteDict) = false
# hasantiderivative(d::ConcreteDict) = false
# hasinterpolationgrid(d::ConcreteDict) = false
# hastransform(d::ConcreteDict) = false
# hastransform(d::ConcreteDict, gb::GridBasis) = false
# hasextension(d::ConcreteDict) = false
# ... and then implement those operations one by one and remove the definitions.

zeros(::Type{T}, d::DerivedDict) where {T} = zeros(T, superdict(d))
tocoefficientformat(a, d::DerivedDict) = tocoefficientformat(a, superdict(d))

# Delegation of methods
for op in (:length, :extensionsize, :size, :interpolation_grid,
        :components, :tail, :ordering, :support, :dimensions)
    @eval $op(d::DerivedDict) = $op(superdict(d))
end

# Delegation of methods with an index parameter
for op in (:size, :element, :support)
    @eval $op(d::DerivedDict, i) = $op(superdict(d), i)
end

approx_length(d::DerivedDict, n::Int) = approx_length(superdict(d), n)

mapped_dict(d::HighlySimilarDerivedDict, map) = similardictionary(d, mapped_dict(superdict(d), map))

dict_in_support(d::DerivedDict, i, x) = in_support(superdict(d), i, x)


#########################
# Indexing and iteration
#########################

IndexStyle(dict::DerivedDict) = IndexStyle(superdict(dict))

native_index(dict::DerivedDict, idx) = native_index(superdict(dict), idx)

linear_index(dict::DerivedDict, idxn) = linear_index(superdict(dict), idxn)

eachindex(dict::DerivedDict) = eachindex(superdict(dict))

linearize_coefficients!(d::DerivedDict, coef_linear::Vector, coef_native) =
    linearize_coefficients!(superdict(d), coef_linear, coef_native)

delinearize_coefficients!(d::DerivedDict, coef_native, coef_linear::Vector) =
    delinearize_coefficients!(superdict(d), coef_native, coef_linear)

approximate_native_size(d::DerivedDict, size_l) = approximate_native_size(superdict(d), size_l)

linear_size(d::DerivedDict, size_n) = linear_size(superdict(d), size_n)


unsafe_eval_element(d::DerivedDict, idx, x) = unsafe_eval_element(superdict(d), idx, x)

unsafe_eval_element_derivative(d::DerivedDict, idx, x, order) =
    unsafe_eval_element_derivative(superdict(d), idx, x, order)


#########################
# Wrapping of operators
#########################

for op in (:transform_dict,)
    @eval $op(d::DerivedDict; options...) = $op(superdict(d); options...)
end

for op in (:derivative_dict, :antiderivative_dict)
    @eval $op(Φ::DerivedDict, order; options...) =
        similardictionary(Φ, $op(superdict(Φ), order; options...))
end


for op in (:extension, :restriction)
    @eval $op(::Type{T}, src::DerivedDict, dest::DerivedDict; options...) where {T} =
        wrap_operator(src, dest, $op(T, superdict(src), superdict(dest); options...))
end

function transform_from_grid(T, d1::GridBasis, d2::DerivedDict, grid; options...)
    op = transform_from_grid(T, d1, superdict(d2), grid; options...)
    wrap_operator(d1, d2, op)
end

function transform_to_grid(T, d1::DerivedDict, d2::GridBasis, grid; options...)
    op = transform_to_grid(T, superdict(d1), d2, grid; options...)
    wrap_operator(d1, d2, op)
end


for op in (:differentiation, :antidifferentiation)
    @eval $op(::Type{T}, d1::DerivedDict, d2::DerivedDict, order; options...) where {T} =
        wrap_operator(d1, d2, $op(T, superdict(d1), superdict(d2), order; options...))
end


function evaluation(::Type{T}, dict::DerivedDict, gb::GridBasis, grid; options...) where {T}
    A = evaluation(T, superdict(dict), gb, grid; options...)
    wrap_operator(dict, gb, A)
end

## Printing

show(io::IO, mime::MIME"text/plain", d::DerivedDict) = composite_show(io, mime, d)

Display.object_parentheses(d::DerivedDict) = false
Display.stencil_parentheses(d::DerivedDict) = false
Display.displaystencil(d::DerivedDict) = [modifiersymbol(d), "(", superdict(d), ")"]

#########################
# Concrete dict
#########################

"""
For testing purposes we define a concrete subset of DerivedDict. This set should
pass all interface tests and be functionally equivalent to the underlying set.
"""
struct ConcreteDerivedDict{S,T} <: HighlySimilarDerivedDict{S,T}
    superdict   ::  Dictionary{S,T}
end

# Implementing similardictionary is all it takes.

similardictionary(dict::ConcreteDerivedDict, dict2::Dictionary) = ConcreteDerivedDict(dict2)

modifiersymbol(d::ConcreteDerivedDict) = "alias"
