# composite_index.jl

"""
A CompositeIndexIterator iterates over a sequence of linear indices. Each index
is a tuple, where the first entry refers to the linear index, and the second
entry is the linear index. The Flatten iterator in Base would yield only the
second entry.

For example, the composite indices of (1:5,1:3) are:
(1,1), (1,2), (1,3), (1,4), (1,5), (2,1), (2,2), (2,3).

In contrast, the Flatten iterator in Base would yield:
1, 2, 3, 4, 5, 1, 2, 3
"""
immutable CompositeIndexIterator{L}
    lengths ::  L
end

start(it::CompositeIndexIterator) = (1,1)

function next(it::CompositeIndexIterator, state)
    i = state[1]
    j = state[2]
    if j == it.lengths[i]
        nextstate = (i+1,1)
    else
        nextstate = (i,j+1)
    end
    (state, nextstate)
end

done(it::CompositeIndexIterator, state) = state[1] > length(it.lengths)

length(it::CompositeIndexIterator) = sum(it.lengths)
