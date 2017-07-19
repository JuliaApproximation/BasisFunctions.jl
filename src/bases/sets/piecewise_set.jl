# piecewise_set.jl

"""
A PiecewiseSet has a function set for each piece in a partition. Its representation
is a MultiArray containing the expansions of all sets combined.
"""
struct PiecewiseSet{P <: Partition,SETS,S,T} <: CompositeSet{S,T}
    sets        ::  SETS
    offsets     ::  Array{Int,1}
    partition   ::  P

    function PiecewiseSet{P,SETS,S,T}(sets, partition) where {P <: Partition,SETS,S,T}
        offsets = compute_offsets(sets)
        new(sets, offsets, partition)
    end

end

const PiecewiseSetSpan{A, F <: PiecewiseSet} = Span{A,F}

# Make a PiecewiseSet by scaling one set to each of the elements of the partition
function PiecewiseSet(set::FunctionSet1d, partition::Partition, n = ones(length(partition))*length(set))
    sets = [rescale(resize(set, n[i]), left(partition, i), right(partition, i)) for i in 1:length(partition)]
    PiecewiseSet(sets, partition)
end

# Make a PiecewiseSet from a list of sets and a given partition
function PiecewiseSet(sets, partition::Partition, T = domaintype(sets[1]))
    # Make sure that the sets are an indexable list of FunctionSet's
    @assert indexable_set(sets, FunctionSet)
    # TODO: We should check that the supports of the sets match the partition pieces

    PiecewiseSet{typeof(partition),typeof(sets),T}(sets, partition)
end

# Construct a piecewise set from a list of sets in 1d
function PiecewiseSet(sets::Array{S}) where {S <: FunctionSet1d}
    for i in 1:length(sets)-1
        @assert right(sets[i]) ≈ left(sets[i+1])
    end
    points = map(left, sets)
    push!(points, right(sets[end]))
    partition = PiecewiseInterval(points)
    PiecewiseSet(sets, partition)
end

function PiecewiseSet(set::FunctionSet, partition::Partition)
    sets = [rescale(set, left(partition[i]), right(partition[i])) for i in 1:length(partition)]
    PiecewiseSet(sets, partition)
end

sets(set::PiecewiseSet) = set.sets

partition(set::PiecewiseSet) = set.partition


name(set::PiecewiseSet) = "Piecewise function set"

similar_set(set::PiecewiseSet, sets) = PiecewiseSet(sets, partition(set))

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
getindex(set::PiecewiseSet, i, j) = subset(set, (i,j))

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
        @assert nb_elements(s1) == nb_elements(s2)
        # TODO: improve the type of the array elements below
        BlockDiagonalOperator(AbstractOperator{eltype(s1)}[$op(element(s1,i), element(s2, i), order; options...) for i in 1:nb_elements(s1)], s1, s2)
    end
end

"""
Make a piecewise function set from a one-dimensional function set, by inserting
the point x in the support of the original set. This yields a PiecewiseSet on
the partition [left(set), x, right(set)].

The original set is duplicated and rescaled to the two subintervals.
"""
function split_interval(set::FunctionSet1d, x)
    @assert left(set) < x < right(set)
    points = [left(set), x, right(set)]
    PiecewiseSet(set, PiecewiseInterval(points))
end

split_interval(set::PiecewiseSet, x) = split_interval(set, partition_index(partition(set), x), x)

function split_interval(set::PiecewiseSet, i::Int, x)
    part = partition(set)
    @assert left(part, i) < x < right(part, i)

    part2 = split_interval(part, i, x)
    two_sets = split_interval(element(set, i), x)
    PiecewiseSet(insert_at(elements(set), i, elements(two_sets)), part2)
end

# Compute the coefficients in an expansion that results from splitting the given
# expansion at a point x.
function split_interval_expansion(set::FunctionSet1d, coefficients, x)
    pset = split_interval(set, x)
    z = zeros(pset)
    pset1 = element(pset, 1)
    pset2 = element(pset, 2)
    # We will manually evaluate the current function at the approximation grids
    # of the newly created sets with smaller support, and use those function values
    # to reconstruct the original function on each subinterval.
    A1 = approximation_operator(pset1)
    A2 = approximation_operator(pset2)
    E1 = evaluation_operator(set, grid(src(A1)))
    E2 = evaluation_operator(set, grid(src(A2)))
    z1 = A1*(E1*coefficients)
    z2 = A2*(E2*coefficients)
    coefficients!(z, 1, z1)
    coefficients!(z, 2, z2)
    # We return the new set and its expansion coefficients
    pset, z
end

function split_interval_expansion(set::PiecewiseSet, coefficients::MultiArray, x)
    part = partition(set)
    i = partition_index(part, x)
    set_i = element(set, i)
    coef_i = element(coefficients, i)
    split_set, split_coef = split_interval_expansion(set_i, coef_i, x)
    # We compute the types of the individual sets and their coefficients
    # in a hacky way to help inference further on. TODO: fix, because this
    # violates encapsulation and it assumes homogeneous elements
    S = eltype(set.sets)
    C = eltype(coefficients.arrays)

    # Now we want to replace the i-th set by the two new sets, and same for the coefficients
    # Technicalities arise when i is 1 or i equals the nb_elements of the set
    local sets, coefs
    old_sets = elements(set)
    old_coef = elements(coefficients)
    if i > 1
        if i < nb_elements(set)
            # We retain the old elements before and after the new ones
            sets = S[old_sets[1:i-1]..., element(split_set, 1), element(split_set, 2), old_sets[i+1:end]...]
            coefs = C[old_coef[1:i-1]..., element(split_coef, 1), element(split_coef, 2), old_coef[i+1:end]...]
        else
            # We replace the last element, so no elements come after the new ones
            sets = S[old_sets[1:i-1]..., element(split_set, 1), element(split_set, 2)]
            coefs = C[old_coef[1:i-1]..., element(split_coef, 1), element(split_coef, 2)]
        end
    else
        # Here i==1, so there are no elements before the new ones
        sets = S[element(split_set, 1), element(split_set, 2), old_sets[i+1:end]...]
        coefs = C[element(split_coef, 1), element(split_coef, 2), old_coef[i+1:end]...]
    end
    PiecewiseSet(sets), MultiArray(coefs)
end

function dot(set::PiecewiseSet, f1::Int, f2::Function, nodes::Array=BasisFunctions.native_nodes(set); options...)
    idxn = native_index(set, f1)
    b = set.sets[idxn[1]]

    dot(b, linear_index(b,idxn[2]), f2, clip_and_cut(nodes, left(b), right(b)); options...)
end

function Gram(set::PiecewiseSet; options...)
    BlockDiagonalOperator(AbstractOperator{eltype(set)}[Gram(element(set,i); options...) for i in 1:nb_elements(set)], set, set)
end
