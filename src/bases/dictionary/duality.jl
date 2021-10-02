########################
# Duality
########################

"Supertype of trait types which define a kind of duality."
abstract type DualType end

"Orthogonal dual type."
struct OrthogonalDual <: DualType end

"Return the dictionary dual to the given dictionary with the given measure."
dual(dict::Dictionary, measure = measure(dict); options...) =
    dual(OrthogonalDual(), dict, measure; options...)

# dual(dict::Dictionary, measure::Measure=measure(dict); dualtype=gramdual, options...) =
    # dualtype(dict, measure; options...)

dual(::OrthogonalDual, dict, measure; options...) = gramdual(dict, measure; options...)

gramdual(dict::Dictionary, measure::Measure; options...) =
    default_gramdual(dict, measure; options...)

function default_gramdual(dict::Dictionary, measure::Measure; options...)
    try
        conj(inv(gram(dict, measure; options...))) * dict
    catch DimensionMismatch
        @warn "Convert DictionaryOperator to dense ArrayOperator"
        ArrayOperator(inv(conj(Matrix(gram(dict, measure; options...)))),dict,dict) * dict
    end
end
