# operated_set.jl

"""
An OperatedSet represents a set that is acted on by an operator, for example the differentiation operator.
The OperatedSet is like the destination set of the operator, up to a change of basis.

Vice-versa, OperatedSet's can be used to represent a change of basis.
"""
immutable OperatedSet{S1,S2,OP,T} <: FunctionSet{1,T}
    "The set that is acted on."
    src_set     ::  S1
    "The destination set of the operator."
    dest_set    ::  S2
    "The operator that acts on the set"
    op          ::  OP

    scratch_src  ::  Array{T,1}
    scratch_dest ::  Array{T,1}

    function OperatedSet(src_set, dest_set, op::AbstractOperator)
        scratch_src = zeros(T, length(src_set))
        scratch_dest = zeros(T, length(dest_set))
        new(src_set, dest_set, op, scratch_src, scratch_dest)
    end
end


function OperatedSet(op::AbstractOperator)
    ELT = eltype(op)
    src_set = src(op)
    dest_set = dest(op)
    OperatedSet{typeof(src_set),typeof(dest_set),typeof(op),ELT}(src_set, dest_set, op)
end

name(s::OperatedSet) = name(dest(s)) * " transformed by an operator"

src(s::OperatedSet) = s.src_set

dest(s::OperatedSet) = s.dest_set

operator(s::OperatedSet) = s.op

# TODO: add promote_eltype to operators and then come back to fix this
function promote_eltype{T}(s::OperatedSet, ::Type{T})
    src_set = promote_eltype(s.src_set, T)
    dest_set = promote_eltype(s.dest_set, T)
    OperatedSet{typeof(src_set),typeof(dest_set),typeof(s.op),T}(src_set, dest_set, s.op)
end

# Also need resize for operators
#resize(s::OperatedSet, n) = ?

for op in (:left, :right, :length)
    @eval $op(b::OperatedSet) = $op(dest(b))
end


function call_element(s::OperatedSet, i, x)
    s.scratch_src[i] = 1
    apply!(s.op, s.scratch_dest, s.scratch_src)
    s.scratch_src[i] = 0
    call_expansion(dest(s), s.scratch_dest, x)
end

## Traits
# This can be improved once operators also support isreal etcetera.

for op in (:isreal,)
    @eval $op{S1,S2,OP,T}(::Type{OperatedSet{S1,S2,OP,T}}) = $op(S1) & $op(S2)
end

is_basis{S1,S2,OP,T}(::Type{OperatedSet{S1,S2,OP,T}}) = False
is_frame{S1,S2,OP,T}(::Type{OperatedSet{S1,S2,OP,T}}) = False
is_orthogonal{S1,S2,OP,T}(::Type{OperatedSet{S1,S2,OP,T}}) = False
is_biorthogonal{S1,S2,OP,T}(::Type{OperatedSet{S1,S2,OP,T}}) = False



derivative(s::FunctionSet; options...) = OperatedSet(differentiation_operator(s; options...))


