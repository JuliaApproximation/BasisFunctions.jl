

#######################
## Application support
#######################

"""
Compute the moment of the given basisfunction, i.e. the integral on its
support.
"""
function moment(dict::Dictionary1d, idx; options...)
    @boundscheck checkbounds(dict, idx)
    unsafe_moment(dict, native_index(dict, idx); options...)
end

# This routine is called after the boundscheck. Call another function,
# default moment, so that unsafe_moment of the concrete dictionary can still
# fall back to `default_moment` as well for some values of the index.
unsafe_moment(dict::Dictionary, idx; options...) = default_moment(dict, idx; options...)

# Default to numerical integration
default_moment(dict::Dictionary, idx; measure = lebesguemeasure(support(dict)), options...) =
    innerproduct(dict[idx], x->1, measure; options...)

dict_norm(dict, idx) = dict_norm(dict, idx, measure(dict))

function dict_norm(dict, idx, μ::Measure)
    @boundscheck checkbounds(dict, idx)
    sqrt(innerproduct(dict, idx, dict, idx, μ))
end
