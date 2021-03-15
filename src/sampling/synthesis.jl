
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
    DiscreteVectorDictionary{coefficienttype(dict)}(length(dict))


"""
The synthesis operator maps a discrete set of coefficients to a function
in the span of a dictionary.
"""
struct SynthesisOperator <: AbstractOperator
    dictionary  ::  Dictionary
    measure     ::  Union{Measure,Nothing}
end

SynthesisOperator(dict::Dictionary) = hasmeasure(dict) ?
    SynthesisOperator(dict, measure(dict)) :
    SynthesisOperator(dict, nothing)

dictionary(op::SynthesisOperator) = op.dictionary

src(op::SynthesisOperator) = discrete_set(dictionary(op))
src_space(op::SynthesisOperator) = Span(src(op))
dest_space(op::SynthesisOperator) = Span(dictionary(op))

(op::SynthesisOperator)(args...) = apply(op, args...)

# We attempt to convert the given coefficients to the container type of the dictionary
apply(op::SynthesisOperator, coef; options...) = _apply(op, coef, dictionary(op))
_apply(op::SynthesisOperator, coef, dict::Dictionary) = Expansion(dict, convert(containertype(dict), coef))

apply(op::SynthesisOperator, expansion::Expansion{D}; options...) where {D <: DiscreteDictionary} =
    apply(op, coefficients(expansion); options...)
apply(op::SynthesisOperator, coef::Expansion; opts...) =
    error("A synthesis operator applies only to coefficients or expansions in discrete sets.")


hasstencil(op::SynthesisOperator) = true
stencilarray(op::SynthesisOperator) = stencilarray(op, dictionary(op), op.measure)
stencilarray(op::SynthesisOperator, dict::Dictionary, ::Nothing) =
    [modifiersymbol(op), "[", dict, "]"]
stencilarray(op::SynthesisOperator, dict::Dictionary, measure::Measure) =
    [modifiersymbol(op), "[", dict, ", ", measure, "]"]

modifiersymbol(op::SynthesisOperator) = PrettyPrintSymbol{:𝒯}()#PrettyPrintSymbol{:ℙ}()
name(::PrettyPrintSymbol{:𝒯}) = "Synthesis operator of a dictionary"
name(op::SynthesisOperator) = "Synthesis operator of a dictionary"
strings(op::SynthesisOperator) = strings(op, dictionary(op), op.measure)
strings(op::SynthesisOperator, dict::Dictionary, ::Nothing) = (strings(dict),)
strings(op::SynthesisOperator, dict::Dictionary, measure::Weight) = (strings(dict),strings(measure))


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
apply(op::DesynthesisOperator, expansion::Expansion; opts...) = coefficients(expansion)

(*)(op::DesynthesisOperator, expansion::Expansion) = apply(op, expansion)

inv(op::SynthesisOperator) = DesynthesisOperator(dictionary(op))
inv(op::DesynthesisOperator) = SynthesisOperator(dictionary(op))


"Object that can be indexed with a dictionary to construct a synthesis operator."
struct SynthesisOperatorGenerator
end

getindex(op::SynthesisOperatorGenerator, Φ::Dictionary) = SynthesisOperator(Φ)

# the symbol is \bscrT
export 𝓣
"Generate a synthesis operator of a dictionary by indexing 𝓣 with that dictionary."
const 𝓣 = SynthesisOperatorGenerator()
