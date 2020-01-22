
# Methods for the computation of Gram matrices and continuous projections in general

# By convention Gram functionality is only implemented for dictionaries that are
# associated with a measure.
hasmeasure(dict::Dictionary) = false

# Determine a measure to use when two dictionaries are given
defaultmeasure(dict1::Dictionary, dict2::Dictionary) =
    _defaultmeasure(dict1, dict2, measure(dict1), measure(dict2))

function _defaultmeasure(dict1, dict2, m1, m2)
    if iscompatible(m1, m2)
        m1
    else
        if iscompatible(support(dict1),support(dict2))
            GenericLebesgueMeasure(support(dict1))
        else
            error("Please specify which measure to use for the combination of $(dict1) and $(dict2).")
        end
    end
end

# Shortcut: Dictionaries of the same type have just one measure
defaultmeasure(dict1::D, dict2::D) where {D <: Dictionary} = measure(dict1)


innerproduct(dict1::Dictionary, i, dict2::Dictionary, j; options...) =
    innerproduct(dict1, i, dict2, j, defaultmeasure(dict1, dict2); options...)

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
# - innerproduct1: possibility to dispatch on the first dictionary without ambiguity.
#                  If not specialized, we call innerproduct2
innerproduct1(dict1::Dictionary, i, dict2, j, measure; options...) =
    innerproduct2(dict1, i, dict2, j, measure; options...)
# - innerproduct2: possibility to dispatch on the second dictionary without ambiguity
innerproduct2(dict1, i, dict2::Dictionary, j, measure; options...) =
    default_dict_innerproduct(dict1, i, dict2, j, measure; options...)


# We make this a separate routine so that it can also be called directly, in
# order to compare to the value reported by a dictionary overriding innerproduct
function default_dict_innerproduct(dict1::Dictionary, i, dict2::Dictionary, j, measure; options...)
	@boundscheck checkbounds(dict1, i)
	@boundscheck checkbounds(dict2, i)
    applymeasure(measure, x->conj(unsafe_eval_element(dict1, i, x)) * unsafe_eval_element(dict2, j, x); options...)
end

gramelement(dict::Dictionary, i, j, m = measure(dict); options...) =
    innerproduct(dict, i, dict, j, m; options...)

# Call this routine in order to evaluate the Gram matrix entry numerically
default_gramelement(dict::Dictionary, i, j, m=measure(dict); options...) =
    default_dict_innerproduct(dict, i, dict, j, m; options...)

function grammatrix(dict::Dictionary, m=measure(dict); T=codomaintype(dict),options...)
    G = zeros(T, length(dict), length(dict))
    grammatrix!(G, dict, m; options...)
end

function grammatrix!(G, dict::Dictionary, m=measure(dict); options...)
    n = length(dict)
    for i in 1:n
        for j in 1:i-1
            G[i,j] = gramelement(dict, i, j, m; options...)
            G[j,i] = conj(G[i,j])
        end
        G[i,i] = gramelement(dict, i, i, m; options...)
    end
    G
end

gramoperator(dict::Dictionary; options...) =
    gramoperator(dict, measure(dict); options...)

gramoperator(dict::Dictionary, measure::Measure; options...) =
    gramoperator1(dict, measure; options...)

gramoperator1(dict::Dictionary, measure; options...) =
    gramoperator2(dict, measure; options...)

gramoperator2(dict, measure::Measure; options...) =
    default_gramoperator(dict, measure; options...)

gramoperator2(dict, measure::DiscreteMeasure; options...) =
    gramoperator(dict, measure, grid(measure), weights(measure); options...)

gramoperator(dict, measure::DiscreteMeasure, grid::AbstractGrid, weights; options...) =
    default_mixedgramoperator_discretemeasure(dict, dict, measure, grid, weights; options...)

function default_gramoperator(dict::Dictionary, m::Measure=measure(dict); options...)
    @debug "Slow computation of Gram matrix entrywise."
    A = grammatrix(dict, m; options...)
    R = ArrayOperator(A, dict, dict)
end

function default_diagonal_gramoperator(dict::Dictionary, measure::Measure; T = coefficienttype(dict), options...)
    @assert isorthogonal(dict, measure)
	n = length(dict)
	diag = zeros(T, n)
	for i in 1:n
		diag[i] = innerproduct(dict, i, dict, i, measure; options...)
	end
	DiagonalOperator(dict, diag)
end

function default_mixedgramoperator_discretemeasure(dict1::Dictionary, dict2::Dictionary, measure::DiscreteMeasure, grid::AbstractGrid, weights;
            T = promote_type(op_eltype(dict1,dict2),subdomaintype(measure)), options...)
    E1 = evaluation_operator(dict1, grid; T=T, options...)
    E2 = evaluation_operator(dict2, grid; T=T, options...)
    W = DiagonalOperator(dest(E2),dest(E1), weights, T=T)
    E1'*W*E2
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


# If no measure is given, try to determine a default choice from the measures of
# the given dictionaries. If they agree, we use that one, otherwise we throw an error.
mixedgramoperator(d1::Dictionary, d2::Dictionary; options...) =
    mixedgramoperator(d1, d2, measure(d1), measure(d2); options...)

function mixedgramoperator(d1, d2, m1::Measure, m2::Measure; options...)
    if iscompatible(m1, m2)
        mixedgramoperator(d1, d2, m1; options...)
    else
        error("Incompatible measures: mixed gram operator is ambiguous.")
    end
end

"""
Compute the mixed Gram matrix corresponding to two dictionaries. The matrix
has elements given by the inner products between the elements of the dictionaries,
relative to the given measure.
"""
mixedgramoperator(d1, d2, measure; options...) =
    mixedgramoperator1(d1, d2, measure; options...)

# The routine mixedgramoperator1 can be specialized by concrete subtypes of the
# first dictionary, while mixedgramoperator2 can be specialized on the second dictionary.
# mixedgramoperator3 can be specialized on the measure
mixedgramoperator1(d1::Dictionary, d2, measure; options...) =
    mixedgramoperator2(d1, d2, measure; options...)

mixedgramoperator2(d1, d2::Dictionary, measure; options...) =
    mixedgramoperator3(d1, d2, measure; options...)

mixedgramoperator3(d1, d2, measure::Measure; options...) =
    default_mixedgramoperator(d1, d2, measure; options...)

mixedgramoperator3(d1, d2, measure::DiscreteMeasure; options...) =
    mixedgramoperator(d1, d2, measure, grid(measure), weights(measure); options...)

mixedgramoperator(d1, d2, measure::DiscreteMeasure, grid::AbstractGrid, weights; options...) =
    default_mixedgramoperator_discretemeasure(d1, d2, measure, grid, weights; options...)

function mixedgramoperator(d1::DICT, d2::DICT, measure::Measure; options...) where DICT<:Dictionary
    if d1 == d2
        gramoperator(d1, measure; options...)
    else
        mixedgramoperator1(d1, d2, measure; options...)
    end
end

function default_mixedgramoperator(d1::Dictionary, d2::Dictionary, measure; options...)
    @debug "Slow computation of mixed Gram matrix entrywise."
    A = mixedgrammatrix(d1, d2, measure; options...)
    T = eltype(A)
    ArrayOperator(A, promote_coefficienttype(d2,T), promote_coefficienttype(d1,T))
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
