
@deprecate roots(dict::Dictionary, coef::AbstractVector) expansion_roots(dict, coef)
@deprecate roots(b::OPS) ops_roots(b)

# support of a dictionary
@deprecate support(dict::Dictionary, idx) dict_support(dict, idx)
@deprecate support(part::Partition, idx) partition_support(part, idx)

@deprecate moment(dict::Dictionary1d, idx; options...) dict_moment(dict, idx; options...)


# No longer identify expansions with coefficient vector
# This functionality was removed in v0.7. For documentation purposes we list deprecations here:

# @deprecate iterate(e::Expansion) iterate(coefficients(e))
# @deprecate iterate(e::Expansion, state) iterate(coefficients(e), state)
# @deprecate collect(e::Expansion) coefficients(e)
# @deprecate getindex(e::Expansion, args...) getindex(e.coefficients, args...)
# @deprecate setindex!(e::Expansion, args...) getindex(e.coefficients, args...)

# import Base: BroadcastStyle
# @deprecate BroadcastStyle(e::Expansion) Base.Broadcast.DefaultArrayStyle{dimension(e)}()
