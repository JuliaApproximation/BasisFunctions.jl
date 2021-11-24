
"Supertype of trait types which define a kind of duality."
abstract type DualType end

"Orthogonal dual type: duality with respect to the continuous measure of the dictionary."
struct OrthogonalDual <: DualType end
"Discrete orthogonal dual type: duality with respect to the discrete measure of the dictionary."
struct DiscreteOrthogonalDual <: DualType end
"Duality defined by the inverse of the Gram matrix with a given measure."
struct GramDual <: DualType end

"Return the dictionary dual to `dict` in the sense defined by the dual type."
function dual(::DualType, dict; options...) end

# Use orthogonal dual as the default
dual(dict::Dictionary, args...; options...) = dual(OrthogonalDual(), dict, args...; options...)

dual(::OrthogonalDual, dict; options...) = gramdual(dict, measure(dict); options...)
dual(::DiscreteOrthogonalDual, dict; options...) = gramdual(dict, discretemeasure(dict); options...)
# For GramDual, the measure has to be specified as a keyword argument
dual(::GramDual, dict; measure, options...) = gramdual(dict, measure; options...)

gramdual(dict::Dictionary, measure::Measure; options...) =
    default_gramdual(dict, measure; options...)

function default_gramdual(dict::Dictionary, measure::Measure; options...)
    try
        conj(inv(gram(dict, measure; options...))) * dict
    catch DimensionMismatch
        # @warn "Convert DictionaryOperator to dense ArrayOperator"
        # ArrayOperator(inv(conj(Matrix(gram(dict, measure; options...)))),dict,dict) * dict
        error("When does this happen? Follow the stack trace and fix the underlying issue.")
    end
end
