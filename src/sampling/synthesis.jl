# synthesis.jl

"Which discrete dictionary matches the coefficients of the given dictionary?"
discrete_set(dict::Dictionary) = discrete_set(dict, containertype(dict))
# We dispatch on the container type:
# - it is a vector
discrete_set(dict::Dictionary, ::Type{Vector{T}}) where {T} =
    DiscreteVectorDictionary{T}(length(dict))
# - it is an array
discrete_set(dict::Dictionary, ::Type{Array{T,N}}) where {T,N} =
    DiscreteArrayDictionary{T}(size(dict))
# - it is something else that we don't know about here: return a vector set
discrete_set(dict::Dictionary, ::Type{F}) where {F} =
    DiscreteVectorDictionary{coefficient_type(dict)}(length(dict))


"""
The synthesis operator maps a discrete set of coefficients to a function
in the span of a dictionary.
"""
struct SynthesisOperator <: AbstractOperator
    dictionary  ::  Dictionary
end

dictionary(op::SynthesisOperator) = op.dictionary

src(op::SynthesisOperator) = discrete_set(dictionary(op))
src_space(op::SynthesisOperator) = Span(src(op))
dest_space(op::SynthesisOperator) = Span(dictionary(op))

# We attempt to convert the given coefficients to the container type of the dictionary
apply(op::SynthesisOperator, coef) = _apply(op, coef, dictionary(op))
_apply(op::SynthesisOperator, coef, dict) = Expansion(dict, convert(containertype(dict), coef))

apply(op::SynthesisOperator, expansion::Expansion{D}) where {D <: DiscreteDictionary} =
    apply(op, coefficients(expansion))
apply(op::SynthesisOperator, coef::Expansion) =
    error("A synthesis operator applies only to coefficients or expansions in discrete sets.")

(*)(op::SynthesisOperator, coef) = apply(op, coef)


"""
The desynthesis operator does the inverse of the synthesis operator: it maps
an expansion in a dictionary to an expansion in the corresponding discrete set.
"""
struct DesynthesisOperator <: AbstractOperator
    dictionary  ::  Dictionary
end

dictionary(op::DesynthesisOperator) = op.dictionary

src_space(op::DesynthesisOperator) = Span(dictionary(op))
dest(op::DesynthesisOperator) = discrete_set(dictionary(op))
dest_space(op::DesynthesisOperator) = Span(dest(op))

# We accept any expansion and return its coefficients
apply(op::DesynthesisOperator, expansion::Expansion) = coefficients(expansion)

(*)(op::DesynthesisOperator, expansion::Expansion) = apply(op, expansion)

inv(op::SynthesisOperator) = DesynthesisOperator(dictionary(op))
inv(op::DesynthesisOperator) = SynthesisOperator(dictionary(op))


"""
The analysis operator associated with a dictionary maps a function to the vector
of inner products with the elements of the dictionary.
"""
struct AnalysisOperator{S,T} <: AbstractOperator
    dictionary  ::  Dictionary{S,T}
end

dictionary(op::AnalysisOperator) = op.dictionary

src(op::AnalysisOperator{S,T}) where {S,T} = FunctionSpace{S,T}()
dest(op::AnalysisOperator) = discrete_set(dictionary(op))

apply(op::AnalysisOperator, span::Span; options...) = Gram(D; options...)
