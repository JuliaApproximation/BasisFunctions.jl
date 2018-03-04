# indexing.jl

############
# Overview #
############

# We collect some methods and definitions having to do with various kinds of
# indexing. We make some assumptions on all index sets:
# - an index set is mathematically a set, i.e., it has no duplicates
# - in addition to being a set, the elements can be ordered and the ordering
#   is defined by the iterator of the set.
#
# The functionality in this file includes:
# - bounds checking: see `checkbounds` function
# - conversion between indices of different types
# - efficient iterators

# Dictionaries can be indexed in various ways. We assume that the semantics
# of the index is determined by its type and, moreover, that linear indices
# are always Int's. This means that no other index can have type Int.
LinearIndex = Int

# We fall back to whatever membership function (`in`) is defined for the index
# set `I`.
checkbounds(i::LinearIndex, I) = i ∈ I


########################
# Composite indices
########################


"""
A composite index is an index that consists of multiple indices.
The internal representation is a tuple.
"""
struct CompositeIndex{I}
    idx ::  I
end
