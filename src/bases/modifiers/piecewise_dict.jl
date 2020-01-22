
"""
A `PiecewiseDict` has a dictionary for each piece in a partition. Its representation
is a `BlockVector` containing the expansions of all dictionaries combined.
"""
struct PiecewiseDict{P <: Partition,DICTS,S,T} <: CompositeDict{S,T}
    dicts       ::  DICTS
    offsets     ::  Vector{Int}
    partition   ::  P

    function PiecewiseDict{P,DICTS,S,T}(dicts, partition) where {P <: Partition,DICTS,S,T}
        offsets = compute_offsets(dicts)
        new(dicts, offsets, partition)
    end
end



# Make a PiecewiseDict by scaling one set to each of the elements of the partition
function PiecewiseDict(set::Dictionary1d, partition::Partition, n = ones(Int,length(partition))*length(set))
    dicts = [rescale(resize(set, n[i]), support(partition, i)) for i in 1:length(partition)]
    PiecewiseDict(dicts, partition)
end

# Make a PiecewiseDict from a list of dicts and a given partition
function PiecewiseDict(dicts, partition::Partition)
    S = reduce(promote_type, map(domaintype, dicts))
    T = reduce(promote_type, map(codomaintype, dicts))
    # Make sure that the dicts are an indexable list of Dictionary's
    @assert indexable_list(dicts, Dictionary)
    # TODO: We should check that the supports of the dicts match the partition pieces

    PiecewiseDict{typeof(partition),typeof(dicts),S,T}(dicts, partition)
end

# Construct a piecewise set from a list of dicts in 1d
function PiecewiseDict(dicts::Array{S}) where {S <: Dictionary1d}
    # Is this not overly restrictive? A redefinition of the support of a piecewise set could allow disjointed intervals.
    for i in 1:length(dicts)-1
        @assert supremum(support(dicts[i])) ≈ infimum(support(dicts[i+1]))
    end
    points = map(i->infimum(support(i)), dicts)
    push!(points, supremum(support(dicts[end])))
    partition = PiecewiseInterval(points)
    PiecewiseDict(dicts, partition)
end

function PiecewiseDict(set::Dictionary, partition::Partition)
     dicts = [rescale(set, support(partition, i)) for i in 1:length(partition)]
    PiecewiseDict(dicts, partition)
end

name(dict::PiecewiseDict) = "Piecewise dictionary"

dictionaries(set::PiecewiseDict) = set.dicts

partition(set::PiecewiseDict) = set.partition

iscomposite(set::PiecewiseDict) = true

function stencilarray(dict::PiecewiseDict)
    A = Any[]
    push!(A, "(")
    i = 1
    for s in dictionaries(dict)
        i != 1 && push!(A, ", ")
        push!(A, s)
        i += 1
    end
    push!(A, ")")
    A
end



similar_dictionary(set::PiecewiseDict, dicts) = PiecewiseDict(dicts, partition(set))

# The set is orthogonal, biorthogonal, etcetera, if all its subsets are.
for op in (:isbasis, :isframe)
    @eval $op(s::PiecewiseDict) = reduce(&, map($op, elements(s)))
end

# The set is orthogonal, biorthogonal, etcetera, if all its subsets are.
for op in (:isorthogonal, :isbiorthogonal, :isbasis, :isframe)
    @eval $op(s::PiecewiseDict, m::Measure) =
        (@warn "definition unclear";reduce(&, map(x->$op(x, m), elements(s))))# or take intersection of measure and support of dictpiece
end
isorthonormal(s::PiecewiseDict, m::Measure) = false

for op in (:support,)
    @eval $op(set::PiecewiseDict) = $op(partition(set))
    @eval $op(set::PiecewiseDict, idx::Int) = $op(set, multilinear_index(set, idx))
    @eval $op(set::PiecewiseDict, idx) = $op(element(set, idx[1]), idx[2])
end

# The set has a grid and a transform if all its subsets have it
# Disable for now, until grids can be collected into a MultiGrid or something
#for op in (:hasinterpolationgrid, :hastransform)
#    @eval $op(s::PiecewiseDict) = reduce(&, map($op, elements(s)))
#end

# We have to override getindex for CompositeDict's, because getindex for a
# CompositeDict calls getindex on one of the underlying sets. However, calling
# a function of the underlying set currently does not return zero for an argument
# that is outside its support. In order to get that behaviour, we need to retain
# the PiecewiseDict.
# Perhaps this should change, and any function should be zero outside its support.
getindex(set::PiecewiseDict, i, j) = subdict(set, (i,j))

function unsafe_eval_element(set::PiecewiseDict, idx::Tuple{Int,Any}, x)
    x ∈ set.partition[idx[1]] ? unsafe_eval_element( element(set, idx[1]), idx[2], x) : zero(eltype(x))
end

function eval_expansion(set::PiecewiseDict, x)
    i = partition_index(set, x)
    eval_expansion(element(set, i), x)
end

# TODO: improve, by subdividing the given grid according to the subregions of the piecewise set
grid_evaluation_operator(dict::PiecewiseDict, gb::GridBasis, grid::AbstractGrid; T=op_eltype(dict,gb), options...) =
    ArrayOperator(evaluation_matrix(dict, grid; T=T), dict, gb) * LinearizationOperator(dict; T=T)


for op in [:differentiation_operator, :antidifferentiation_operator]
    @eval function $op(s1::PiecewiseDict, s2::PiecewiseDict, order; T=op_eltype(s1,s2), options...)
        @assert numelements(s1) == numelements(s2)
        # TODO: improve the type of the array elements below
        BlockDiagonalOperator(DictionaryOperator{T}[$op(element(s1,i), element(s2, i), order; options...) for i in 1:numelements(s1)], s1, s2; T=T)
    end
end

"""
Make a piecewise function set from a one-dimensional function set, by inserting
the point x in the support of the original set. This yields a PiecewiseDict on
the partition [left(set), x, right(set)].

The original set is duplicated and rescaled to the two subintervals.
"""
function split_interval(set::Dictionary1d, x)
    @assert infimum(support(set)) < x < supremum(support(set))
    points = [infimum(support(set)), x, supremum(support(set))]
    PiecewiseDict(set, PiecewiseInterval(points))
end

split_interval(set::PiecewiseDict, x) = split_interval(set, partition_index(partition(set), x), x)

function split_interval(set::PiecewiseDict, i::Int, x)
    part = partition(set)
    @assert infimum(support(part, i)) < x < supremum(support(part, i))

    part2 = split_interval(part, i, x)
    two_sets = split_interval(element(set, i), x)
    PiecewiseDict(insert_at(elements(set), i, elements(two_sets)), part2)
end

# Compute the coefficients in an expansion that results from splitting the given
# expansion at a point x.
function split_interval_expansion(set::Dictionary1d, coefficients, x)
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

function split_interval_expansion(set::PiecewiseDict, coefficients::BlockVector, x)
    part = partition(set)
    i = partition_index(part, x)
    set_i = element(set, i)
    coef_i = getblock(coefficients, i)
    split_set, split_coef = split_interval_expansion(set_i, coef_i, x)
    # We compute the types of the individual sets and their coefficients
    # in a hacky way to help inference further on. TODO: fix, because this
    # violates encapsulation and it assumes homogeneous elements
    S = eltype(set.dicts)
    C = eltype(coefficients)

    # Now we want to replace the i-th set by the two new sets, and same for the coefficients
    # Technicalities arise when i is 1 or i equals the numelements of the set
    local dicts, coefs
    old_dicts = elements(set)
    old_coef = elements(coefficients)
    if i > 1
        if i < numelements(set)
            # We retain the old elements before and after the new ones
            dicts = S[old_dicts[1:i-1]..., element(split_set, 1), element(split_set, 2), old_dicts[i+1:end]...]
            coefs = C[old_coef[1:i-1]..., element(split_coef, 1), element(split_coef, 2), old_coef[i+1:end]...]
        else
            # We replace the last element, so no elements come after the new ones
            dicts = S[old_dicts[1:i-1]..., element(split_set, 1), element(split_set, 2)]
            coefs = C[old_coef[1:i-1]..., element(split_coef, 1), element(split_coef, 2)]
        end
    else
        # Here i==1, so there are no elements before the new ones
        dicts = S[element(split_set, 1), element(split_set, 2), old_dicts[i+1:end]...]
        coefs = C[element(split_coef, 1), element(split_coef, 2), old_coef[i+1:end]...]
    end
    PiecewiseDict(dicts), BlockVector(coefs)
end

# TODO: add the measure argument here
gramoperator(dict::PiecewiseDict; T=coefficienttype(dict), options...) =
    BlockDiagonalOperator(DictionaryOperator{T}[gramoperator(element(dict,i); options...) for i in 1:numelements(dict)], dict, dict; T=T)
