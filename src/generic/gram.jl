
# Methods for the computation of Gram matrices and continuous projections in general

# By convention Gram functionality is only implemented for dictionaries that are
# associated with a measure.
has_measure(dict::Dictionary) = false

gramelement(dict::Dictionary, i, j; options...) =
    innerproduct(dict, i, dict, j; options...)

innerproduct(dict1::Dictionary, i, dict2::Dictionary, j; options...) =
    innerproduct(dict1, i, dict2, j, measure(dict1); options...)

# Convert linear indices to native indices, then call innerproduct_native
innerproduct(dict1::Dictionary, i::Int, dict2::Dictionary, j::Int, measure; options...) =
    innerproduct_native(dict1, native_index(dict1, i), dict2, native_index(dict2, j), measure; options...)
innerproduct(dict1::Dictionary, i, dict2::Dictionary, j::Int, measure; options...) =
    innerproduct_native(dict1, i, dict2, native_index(dict2, j), measure; options...)
innerproduct(dict1::Dictionary, i::Int, dict2::Dictionary, j, measure; options...) =
    innerproduct_native(dict1, native_index(dict1, i), dict2, j, measure; options...)
innerproduct(dict1::Dictionary, i, dict2::Dictionary, j, measure; options...) =
    innerproduct_native(dict1, i, dict2, j, measure; options...)

# - innerproduct_native: if not specialized, called innerproduct1
innerproduct_native(dict1::Dictionary, i, dict2::Dictionary, j, measure; options...) =
    innerproduct1(dict1, i,  dict2, j, measure; options...)
# - innerproduct1: possibility to dispatch on the first dictionary without amibiguity.
#                  If not specialized, we call innerproduct2
innerproduct1(dict1::Dictionary, i, dict2, j, measure; options...) =
    innerproduct2(dict1, i, dict2, j, measure; options...)
# - innerproduct2: possibility to dispatch on the second dictionary without amibiguity
innerproduct2(dict1, i, dict2::Dictionary, j, measure; options...) =
    default_dict_innerproduct(dict1, i, dict2, j, measure; options...)


# We make this a separate routine so that it can also be called directly, in
# order to compare to the value reported by a dictionary overriding innerproduct
function default_dict_innerproduct(dict1::Dictionary, i, dict2::Dictionary, j, m = measure(dict1);
            warnslow = true, options...)
    warnslow && @warn "Evaluating inner product numerically: $dict1"
    integral(x->conj(unsafe_eval_element(dict1, i, x)) * unsafe_eval_element(dict2, j, x), m; options...)
end

# Call this routine in order to evaluate the Gram matrix entry numerically
default_gramelement(dict::Dictionary, i, j; options...) =
    default_dict_innerproduct(dict, i, dict, j; options...)

function grammatrix(dict::Dictionary; options...)
    G = zeros(codomaintype(dict), length(dict), length(dict))
    grammatrix!(G, dict; options...)
end

function grammatrix!(G, dict::Dictionary; options...)
    n = length(dict)
    for i in 1:n
        for j in 1:i-1
            G[i,j] = gramelement(dict, i, j; options...)
            G[j,i] = conj(G[i,j])
        end
        G[i,i] = gramelement(dict, i, i; options...)
    end
    G
end

function gramoperator(dict::Dictionary; warnslow = BF_WARNSLOW, options...)
    warnslow && @warn "Slow computation of Gram matrix entrywise."
    A = grammatrix(dict; options...)
    MatrixOperator(dict, dict, A)
end


"""
Project the function onto the space spanned by the given dictionary.
"""
project(dict::Dictionary, f, m = measure(dict); T = coefficienttype(dict), options...) =
    project!(zeros(T,dict), dict, f, m; options...)

function project!(result, dict, f, measure; options...)
    for i in eachindex(result)
        result[i] = innerproduct(dict[i], f, measure; options...)
    end
    result
end



########################
# Mixed gram operators
########################

mixedgramoperator(d1::Dictionary, d2::Dictionary) = _mixedgramoperator(d1, d2, measure(d1), measure(d2))

iscompatible(m1::M, m2::M) where {M <: Measure} = m1==m2
iscompatible(m1::Measure, m2::Measure) = false

function _mixedgramoperator(d1, d2, m1::Measure, m2::Measure)
    if iscompatible(m1, m2)
        mixedgramoperator(d1, d2, m1)
    else
        error("Incompatible measures: mixed gram operator is ambiguous.")
    end
end

mixedgramoperator(d1, d2, measure) = mixedgramoperator1(d1, d2, measure)

mixedgramoperator1(d1::Dictionary, d2, measure) = mixedgramoperator2(d1, d2, measure)
mixedgramoperator2(d1, d2::Dictionary, measure) = default_mixedgram(d1, d2, measure)

function default_mixedgram(d1::Dictionary, d2::Dictionary, measure; warnslow = BF_WARNSLOW, options...)
    warnslow && @warn "Slow computation of mixed Gram matrix entrywise."
    A = mixedgrammatrix(d1, d2, measure; options...)
    T = eltype(A)
    MatrixOperator(promote_coefficienttype(d1,T), promote_coefficienttype(d2,T), A)
end

function mixedgrammatrix(d1::Dictionary, d2::Dictionary, measure; options...)
    T = promote_type(coefficienttype(d1),coefficienttype(d2))
    G = zeros(T, length(d1), length(d2))
    mixedgrammatrix!(G, d1, d2, measure; options...)
end

function mixedgrammatrix!(G, d1::Dictionary, d2::Dictionary, measure; options...)
    m = length(d1)
    n = length(d2)
    for i in 1:m
        for j in 1:n
            G[i,j] = innerproduct(d1, i, d2, j, measure; options...)
        end
    end
    G
end
