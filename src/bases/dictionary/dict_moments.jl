

#######################
## Application support
#######################

"""
Compute the moment of the given basisfunction, i.e. the integral on its
support.
"""
function dict_moment(dict, idx; options...)
    checkdictionary(dict)
    @boundscheck checkbounds(dict, idx)
    unsafe_dict_moment(dict, native_index(dict, idx); options...)
end

# This routine is called after the boundscheck.
unsafe_dict_moment(dict::Dictionary, idx; options...) = default_dict_moment(dict, idx; options...)

# Default to numerical integration
default_dict_moment(dict::Dictionary, idx; measure = lebesguemeasure(support(dict)), options...) =
    innerproduct(dict[idx], x->1, measure; options...)


"""
    dict_norm(dict, idx[, μ])

Compute the norm of the given basisfunction of the given dictionary,
with respect to the given measure.
"""
function dict_norm(dict, idx, μ::Measure)
    checkdictionary(dict)
    @boundscheck checkbounds(dict, idx)
    sqrt(innerproduct(dict, idx, dict, idx, μ))
end
dict_norm(dict, idx) = dict_norm(dict, idx, measure(dict))
