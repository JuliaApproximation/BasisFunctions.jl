
@deprecate roots(dict::Dictionary, coef::AbstractVector) expansion_roots(dict, coef)
@deprecate roots(b::OPS) ops_roots(b)

# No longer identify expansions with coefficient vector
@deprecate iterate(e::Expansion) iterate(coefficients(e))
@deprecate iterate(e::Expansion, state) iterate(coefficients(e), state)
import Base: collect
collect(e::Expansion) = error("you didn't!")
# @deprecate collect(e::Expansion) coefficients(e)

# Deprecate the below right away because it conflicts with other ways of computing
# with expansions
# import Base: BroadcastStyle
# @deprecate BroadcastStyle(e::Expansion) Base.Broadcast.DefaultArrayStyle{dimension(e)}()

getindex(e::Expansion, args...) = error("Oh no you did not")
# @deprecate getindex(e::Expansion, args...) getindex(e.coefficients, args...)
@deprecate setindex!(e::Expansion, args...) getindex(e.coefficients, args...)

# support of a dictionary
@deprecate support(dict::Dictionary, idx) dict_support(dict, idx)
@deprecate support(part::Partition, idx) partition_support(part, idx)

@deprecate moment(dict::Dictionary1d, idx; options...) dict_moment(dict, idx; options...)
