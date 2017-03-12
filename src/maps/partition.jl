# partition.jl

"""
A partition divides a region into several subregions. The main algorithm of
every partition is deciding in which subregion a given point x lies, implemented
by the `partition_index` function.

The intent of a partition is to be combined with a function set on each subregion,
effectively creating a piecewise function.

The pieces of a partition have an index ranging from 1 to length(partition).
"""
abstract Partition

"""
A PartitionPiece represents a piece of a partition - it essentially combines
a partition and an index into that partition. It is constructed by indexing
a partition.
"""
immutable PartitionPiece{P <: Partition,I}
    part    ::  P
    idx     ::  I
end

partition(piece::PartitionPiece) = piece.part
index(piece::PartitionPiece) = piece.idx

left(piece::PartitionPiece) = left(partition(piece), index(piece))
right(piece::PartitionPiece) = right(partition(piece), index(piece))

Base.in(x, piece::PartitionPiece) = in_partition(partition(piece), index(piece), x)



function checkbounds(part::Partition, i::Int)
    1 <= i <= length(part) || throw(BoundsError())
end

function getindex(part::Partition, i)
    checkbounds(part, i)
    PartitionPiece(part, i)
end

eachindex(part::Partition) = 1:length(part)


"""
A PiecewiseInterval is a subdivision of an interval into a number of subintervals.
The subintervals are determined by a set of breakpoints ``t_i``, ``i=0,...,n``. The main
interval is ``[t_0,t_n]``. The subintervals always contain the right endpoint, i.e.
the first subinterval is ``[t_0,t_1]``, the second is the halfopen interval
``(t_1,t_2]`` and so on until the n-th interval ``(t_{n-1},t_n]``.
"""
immutable PiecewiseInterval{T} <: Partition
    points  ::  Array{T,1}
end

# n partitions of equal size on the interval [a,b]
PiecewiseInterval(a, b, n::Int) = PiecewiseInterval(collect(linspace(a,b,n+1)))

length(part::PiecewiseInterval) = length(part.points)-1

left(part::PiecewiseInterval) = part.points[1]

left(part::PiecewiseInterval, i::Int) = part.points[i]

right(part::PiecewiseInterval) = part.points[end]

right(part::PiecewiseInterval, i::Int) = part.points[i+1]

"Return the index of the region in which the given point x lies."
function partition_index(part::PiecewiseInterval, x)
    @assert left(part) <= x <= right(part)

    idx = 1
    while x > part.points[idx+1]
        idx += 1
    end
    idx
end

function in_partition(part::PiecewiseInterval, i, x)
    # Not sure this is a good idea, but let's respect the convention that the
    # interval does not contain the left endpoint, unless it is the first interval.
    if i == 1
        left(part, 1) <= x <= right(part, 1)
    else
        left(part, i) < x <= right(part, i)
    end
end

# TODO: applying partition_index repeatedly, say for all points in a grid, could
# be done more efficiently if we can exploit ordering of the points in the grid.
# Perhaps partition_index applied to an iterable set of points could return an
# iterable set of indices?

split_interval(part::PiecewiseInterval, i::Int, x) = PiecewiseInterval([part.points[1:i]...,x,part.points[i+1:end]...])

split_interval(part::PiecewiseInterval, x) = split(part, partition_index(part, x), x)
