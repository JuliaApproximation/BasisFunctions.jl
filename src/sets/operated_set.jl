# operated_set.jl

"""
An OperatedSet represents a set that is acted on by an operator, for example the differentiation operator.
The OperatedSet has the dimension of the source set of the operator, but each basis function
is acted on by the operator.
"""
immutable OperatedSet{T} <: FunctionSet{1,T}
    "The operator that acts on the set"
    op          ::  AbstractOperator{T}

    scratch_src  ::  Array{T,1}
    scratch_dest ::  Array{T,1}

    function OperatedSet(op::AbstractOperator)
        scratch_src = zeros(T, length(src(op)))
        scratch_dest = zeros(T, length(dest(op)))
        new(op, scratch_src, scratch_dest)
    end
end


OperatedSet{ELT}(op::AbstractOperator{ELT}) =
    OperatedSet{ELT}(op)

name(s::OperatedSet) = name(dest(s)) * " transformed by an operator"

src(s::OperatedSet) = src(s.op)

dest(s::OperatedSet) = dest(s.op)

operator(s::OperatedSet) = s.op

# TODO: add promote_eltype to operators and then come back to fix this
function promote_eltype{T}(s::OperatedSet, ::Type{T})
    s
end

for op in (:left, :right, :length)
    @eval $op(b::OperatedSet) = $op(src(b))
end


function call_element(s::OperatedSet, i, x)
    s.scratch_src[i] = 1
    apply!(s.op, s.scratch_dest, s.scratch_src)
    s.scratch_src[i] = 0
    call_expansion(dest(s), s.scratch_dest, x)
end

## Properties

isreal(op::OperatedSet) = isreal(op)

for op in (:is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
    @eval $op(op::OperatedSet) = false
end

derivative(s::FunctionSet; options...) = OperatedSet(differentiation_operator(s; options...))
