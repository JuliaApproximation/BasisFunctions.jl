# piecewise_set.jl

"""
A PiecewiseSet has a function set for each piece in a partition. Its representation
is a MultiArray containing the expansions of all sets combined.
"""
immutable PiecewiseSet{P <: Partition,SETS,N,T} <: CompositeSet{N,T}
    sets        ::  SETS
    offsets     ::  Array{Int,1}
    partition   ::  P

    function PiecewiseSet(sets, partition)
        offsets = compute_offsets(sets)
        new(sets, offsets, partition)
    end
end

function PiecewiseSet(sets, partition, T = eltype(sets[1]), N = ndims(sets[1]))
    PiecewiseSet{typeof(partition),typeof(sets),N,T}(sets, partition)
end

# Construct a piecewise set from a list of sets in 1d
function PiecewiseSet{S <: FunctionSet1d}(sets::Array{S})
    for i in 1:length(sets)-1
        @assert right(sets[i]) ≃ left(sets[i+1])
    end
    points = map(left, sets)
    push!(points, right(sets[end]))
    partition = IntervalPartition(points)
    PiecewiseSet(sets, partition)
end

function PiecewiseSet(set::FunctionSet, partition::Partition)
    sets = [rescale(set, left(partition[i]), right(partition[i])) for i in 1:length(partition)]
    PiecewiseSet(sets, partition)
end

sets(set::PiecewiseSet) = set.sets

partition(set::PiecewiseSet) = set.partition


name(set::PiecewiseSet) = "Piecewise function set"

similar_set(set::PiecewiseSet, sets, T = eltype(set)) = PiecewiseSet(sets, partition(set), T)

# The set is orthogonal, biorthogonal, etcetera, if all its subsets are.
for op in (:is_orthogonal, :is_biorthogonal, :is_basis, :is_frame)
    @eval $op(s::PiecewiseSet) = reduce(&, map($op, elements(s)))
end

for op in (:left, :right)
    @eval $op(set::PiecewiseSet) = $op(partition(set))
    @eval $op(set::PiecewiseSet, idx::Int) = $op(set, multilinear_index(set, idx))
    @eval $op(set::PiecewiseSet, idx) = $op(element(set, idx[1]), idx[2])
end

# The set has a grid and a transform if all its subsets have it
# Disable for now, until grids can be collected into a MultiGrid or something
#for op in (:has_grid, :has_transform)
#    @eval $op(s::PiecewiseSet) = reduce(&, map($op, elements(s)))
#end

# We have to override getindex for CompositeSet's, because getindex for a
# CompositeSet calls getindex on one of the underlying sets. However, calling
# a function of the underlying set currently does not return zero for an argument
# that is outside its support. In order to get that behaviour, we need to retain
# the PiecewiseSet.
# Perhaps this should change, and any function should be zero outside its support.
getindex(set::PiecewiseSet, i, j) = FunctionSubSet(set, (i,j))

function eval_element(set::PiecewiseSet, idx::Tuple{Int,Any}, x)
    x ∈ set.partition[idx[1]] ? eval_element( element(set, idx[1]), idx[2], x) : zero(eltype(x))
end

function eval_expansion(set::PiecewiseSet, x)
    i = partition_index(set, x)
    eval_expansion(element(set, i), x)
end

# TODO: improve, by subdividing the given grid according to the subregions of the piecewise set
evaluation_operator(set::PiecewiseSet, dgs::DiscreteGridSpace; options...) =
    MultiplicationOperator(set, dgs, evaluation_matrix(set, grid(dgs))) *
    LinearizationOperator(set)


for op in [:differentiation_operator, :antidifferentiation_operator]
    @eval function $op(s1::PiecewiseSet, s2::PiecewiseSet, order; options...)
        @assert composite_length(s1) == composite_length(s2)
        # TODO: improve the type of the array elements below
        BlockDiagonalOperator(AbstractOperator{eltype(s1)}[$op(element(s1,i), element(s2, i), order; options...) for i in 1:composite_length(s1)], s1, s2)
    end
end

function dot(set::PiecewiseSet, f1::Int, f2::Function, nodes::Array=BasisFunctions.native_nodes(set); options...)
  idxn = native_index(set, f1)
  b = set.sets[idxn[1]]
  shifted = map(x->max(x, left(b)), nodes)
  shifted = map(x->min(x, right(b)), shifted)
  dot(b, linear_index(b,idxn[2]), f2, shifted; options...)
end

function Gram(set::PiecewiseSet; options...)
  BlockDiagonalOperator(AbstractOperator{eltype(set)}[Gram(element(set,i); options...) for i in 1:composite_length(set)], set, set)
end
