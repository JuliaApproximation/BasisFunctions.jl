
"""
An `OperatedDict` represents a set that is acted on by an operator, for example
the differentiation operator. The `OperatedDict` has the dimension of the source
set of the operator, but each basis function is acted on by the operator.
"""
struct OperatedDict{S,T} <: DerivedDict{S,T}#<: Dictionary{S,T}#
    "The operator that acts on the set"
    op          ::  DictionaryOperator{T}

    scratch_src
    scratch_dest

    function OperatedDict{S,T}(op::DictionaryOperator) where {S,T}
        scratch_src = zeros(src(op))
        scratch_dest = zeros(dest(op))
        new(op, scratch_src, scratch_dest)
    end
end



# TODO: OperatedDict should really be a DerivedDict, deriving from src(op)
has_derivative(s::OperatedDict) = false
has_antiderivative(s::OperatedDict) = false
# has_grid(s::ConcreteSet) = false
has_transform(s::OperatedDict) = false
has_transform(s::OperatedDict, dgs::GridBasis) = false
has_extension(s::OperatedDict) = false

is_basis(::OperatedDict) = false

superdict(dict::OperatedDict) = src(dict)

function OperatedDict(op::DictionaryOperator{T}) where {T}
    S = domaintype(src(op))
    OperatedDict{S,T}(op)
end

has_stencil(s::OperatedDict) = true
function stencil(s::OperatedDict,S)
    A = Any[]
    push!(A,operator(s))
    push!(A," * ")
    push!(A,src(s))
    return recurse_stencil(s,A,S)
end
myLeaves(s::OperatedDict) = (operator(s),myLeaves(src(s))...)

src(s::OperatedDict) = src(s.op)
src_dictionary(s::OperatedDict) = src(s)

dest(s::OperatedDict) = dest(s.op)
dest_dictionary(s::OperatedDict) = dest(s)

domaintype(s::OperatedDict) = domaintype(src_dictionary(s))

operator(set::OperatedDict) = set.op

dict_promote_domaintype(s::OperatedDict{T}, ::Type{S}) where {S,T} =
    OperatedDict(similar_operator(operator(s), promote_domaintype(src(s), S), dest(s) ) )

for op in (:support, :length)
    @eval $op(s::OperatedDict) = $op(src_dictionary(s))
end

# We don't know in general what the support of a specific basis functions is.
# The safe option is to return the support of the set itself for each element.
for op in (:support,)
    @eval $op(s::OperatedDict, idx) = $op(src_dictionary(s))
end

dict_in_support(set::OperatedDict, i, x) = in_support(superdict(set), x)


zeros(::Type{T}, s::OperatedDict) where {T} = zeros(T, src_dictionary(s))

##########################
# Indexing and evaluation
##########################

ordering(dict::OperatedDict) = ordering(src_dictionary(dict))

unsafe_eval_element(dict::OperatedDict, i, x) =
    _unsafe_eval_element(dict, i, x, operator(dict), dict.scratch_src, dict.scratch_dest)

function _unsafe_eval_element(dict::OperatedDict, idxn, x, op, scratch_src, scratch_dest)
    idx = linear_index(dict, idxn)
    if is_diagonal(op)
        diagonal(op, idx) * unsafe_eval_element(src_dictionary(dict), idxn, x)
    else
        scratch_src[idx] = 1
        apply!(op, scratch_dest, scratch_src)
        scratch_src[idx] = 0
        eval_expansion(dest_dictionary(dict), scratch_dest, x)
    end
end

function _unsafe_eval_element(dict::OperatedDict, idxn, x, op::ScalingOperator, scratch_src, scratch_dest)
    idx = linear_index(dict, idxn)
    diagonal(op, idx) * unsafe_eval_element(src_dictionary(dict), idxn, x)
end

grid_evaluation_operator(dict::OperatedDict, dgs::GridBasis, grid::AbstractGrid; options...) =
    WrappedOperator(dict, dgs, grid_evaluation_operator(src(dict), dgs, grid; options...)*operator(dict))

grid_evaluation_operator(dict::OperatedDict, dgs::GridBasis, grid::AbstractSubGrid; options...) =
    WrappedOperator(dict, dgs, grid_evaluation_operator(src(dict), dgs, grid; options...)*operator(dict))

## Properties

isreal(dict::OperatedDict) = isreal(operator(dict))

# Disable for now
# for op in (:is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
#     @eval $op(set::OperatedDict) = is_diagonal(operator(set)) && $op(src(set))
# end


#################
# Special cases
#################

# If a set has a differentiation operator, then we can represent the set of derivatives
# by an OperatedDict.
derivative(dict::Dictionary; options...) = OperatedDict(differentiation_operator(s; options...))

function (*)(a::Number, s::Dictionary)
    T = promote_type(typeof(a), coeftype(s))
    OperatedDict(ScalingOperator(s, convert(T, a)))
end

function (*)(op::DictionaryOperator, s::Dictionary)
    @assert src(op) == s
    OperatedDict(op)
end

function grid_evaluation_operator(s::D, dgs::GridBasis, grid::ProductGrid;
        options...) where {D<: TensorProductDict{N,DT,S,T} where {N,DT <: NTuple{N,BasisFunctions.OperatedDict} where N,S,T}}
    tensorproduct([BasisFunctions.grid_evaluation_operator(si, dgsi, gi; options...) for (si, dgsi, gi) in zip(elements(s), elements(dgs), elements(grid))]...)
end
