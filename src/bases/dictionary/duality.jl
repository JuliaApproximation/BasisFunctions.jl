########################
# Duality
########################

dual(dict::Dictionary, measure::BasisFunctions.Measure=measure(dict); dualtype=gramdual, options...) =
    dualtype(dict, measure; options...)

gramdual(dict::Dictionary, measure::Measure; options...) =
    default_gramdual(dict, measure; options...)

function default_gramdual(dict::Dictionary, measure::Measure; options...)
    try
        conj(inv(gramoperator(dict, measure; options...))) * dict
    catch DimensionMismatch
        @warn "Convert DictionaryOperator to dense ArrayOperator"
        ArrayOperator(inv(conj(Matrix(gramoperator(dict, measure; options...)))),dict,dict) * dict
    end
end
