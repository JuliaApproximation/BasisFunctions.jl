
# Methods for the computation of Gram matrices and continuous projections in general

# By convention Gram functionality is only implemented for dictionaries that are
# associated with a measure.
hasmeasure(dict::Dictionary) = false

gramelement(dict::Dictionary, i::LinearIndex, j::LinearIndex; options...) =
    gramelement(dict, native_index(dict, i), native_index(dict, j); options...)

gramelement(dict::Dictionary, i, j; options...) =
    innerproduct(dict, i, dict, j, measure(dict); options...)

innerproduct(dict1::Dictionary, i, dict2::Dictionary, j; options...) =
    innerproduct(dict1, i, dict2, j, measure(dict1); options...)

# Convert linear indices to native indices, then call default
innerproduct(dict1::Dictionary, i, dict2::Dictionary, j, measure; options...) =
    innerproduct1(dict1, i, dict2, j, measure; options...)
#   _innerproduct(dict1, i, dict2, j, measure; options...)
# _innerproduct(dict1::Dictionary, i::LinearIndex, dict2::Dictionary, j::LinearIndex, measure; options...) =
#     innerproduct(dict1, native_index(dict1, i), dict2, native_index(dict2, j), measure; options...)
# _innerproduct(dict1::Dictionary, i, dict2::Dictionary, j::LinearIndex, measure; options...) =
#     innerproduct(dict1, i, dict2, native_index(dict2, j), measure; options...)
# _innerproduct(dict1::Dictionary, i::LinearIndex, dict2::Dictionary, j, measure; options...) =
#     innerproduct(dict1, native_index(dict1, i), dict2, j, measure; options...)
# _innerproduct(dict1::Dictionary, i, dict2::Dictionary, j, measure; options...) =
#     default_dict_innerproduct(dict1, i, dict2, j, measure; options...)

innerproduct1(dict1::Dictionary, i::LinearIndex, dict2, j, measure; options...) =
    innerproduct(dict1, native_index(dict1, i), dict2, j, measure; options...)
innerproduct1(dict1::Dictionary, i, dict2, j, measure; options...) =
    innerproduct2(dict1, i, dict2, j, measure; options...)
innerproduct2(dict1, i, dict2::Dictionary, j::LinearIndex, measure; options...) =
    innerproduct(dict1, i, dict2, native_index(dict2, j), measure; options...)
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
default_gramelement(dict::Dictionary, i::LinearIndex, j::LinearIndex; options...) =
    default_gramelement(dict, native_index(dict, i), native_index(dict, j); options...)

default_gramelement(dict::Dictionary, i, j; options...) =
    default_dict_innerproduct(dict, i, dict, j, measure(dict); options...)

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
