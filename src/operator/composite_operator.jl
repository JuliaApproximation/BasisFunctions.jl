# composite_operator.jl

"""
A composite operator consists of a sequence of operators that are applied
consecutively.
"""
immutable CompositeOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}
    "The list of operators"
    ops     ::  Vector{AbstractOperator}
    "Scratch space for the result of each operator, except the last one"
    scratch
    "The number of operators"
    L       ::  Int

    CompositeOperator(ops, scratch) = new(ops, scratch, length(ops))
end

# Generic functions for composite types:
elements(op::CompositeOperator) = op.ops
element(op::CompositeOperator, j::Int) = op.ops[j]
composite_length(op::CompositeOperator) = op.L

src(op::CompositeOperator, j::Int = 1) = src(op.ops[j])
dest(op::CompositeOperator, j::Int = op.L) = dest(op.ops[j])

function CompositeOperator(operators::AbstractOperator...)
    ELT = promote_type(map(eltype, operators)...)
    SRC = typeof(src(operators[1]))
    DEST = typeof(src(operators[end]))
    L = length(operators)
    scratch = tuple([zeros(ELT, size(dest(operators[j]))) for j in 1:L-1]...)
    CompositeOperator{SRC,DEST}([operators...], scratch)
end

apply!(op::CompositeOperator, dest, src, coef_srcdest) =
    apply_composite!(op, op.ops, coef_srcdest)

apply!(op::CompositeOperator, dest, src, coef_dest, coef_src) =
    apply_composite!(op, op.ops, op.scratch, coef_dest, coef_src)


# TODO: provide more efficient implementation that exploits inplaceness
function apply_composite!(op::CompositeOperator, operators, scratch, coef_dest, coef_src)
    L = composite_length(op)
    apply!(operators[1], scratch[1], coef_src)
    for i in 2:L-1
        apply!(operators[i], scratch[i], scratch[i-1])
    end
    apply!(operators[L], coef_dest, scratch[L-1])
end

# TODO: remove the assumption that all intermediate operators are in-place
function apply_composite!(op::CompositeOperator, operators, coef_srcdest)
    for operator in operators
        apply!(operator, coef_srcdest)
    end
end

inv(op::CompositeOperator) = CompositeOperator(map(inv, op.ops)...)

ctranspose(op::CompositeOperator) = CompositeOperator(map(ctranspose, op.ops)...)

compose() = nothing
compose(ops::AbstractOperator...) = CompositeOperator(flatten(CompositeOperator, ops...)...)

(*)(op2::AbstractOperator, op1::AbstractOperator) = compose(op1, op2)
(*)(op3::AbstractOperator, op2::AbstractOperator, op1::AbstractOperator) = compose(op1, op2, op3)
(*)(ops::AbstractOperator...) = compose([ops[i] for i in length(ops):-1:1]...)

"""
A composite operator of exactly two operators.
"""
immutable CompositeOperator2{OP1,OP2,ELT,N,SRC,DEST} <: AbstractOperator{SRC,DEST}
    op1     ::  OP1
    op2     ::  OP2
    scratch ::  Array{ELT,N}    # For storing the intermediate result after applying op1

    function CompositeOperator2(op1::AbstractOperator, op2::AbstractOperator)
        @assert size(op1,1) == size(op2,2)

        # Possible optimization here would be to avoid allocating memory if the second operator is in-place.
        # But even in that case, the user may invoke the operator in a non-in-place way, so let's keep it.
        new(op1, op2, zeros(ELT,size(src(op2))))
    end
end

# We could ask that DEST1 == SRC2 but that might be too strict. As long as the operators are compatible things are fine.
function CompositeOperator2{SRC1,DEST1,SRC2,DEST2}(op1::AbstractOperator{SRC1,DEST1}, op2::AbstractOperator{SRC2,DEST2})
        #@assert DEST1 == SRC2
    OP1 = typeof(op1)
    OP2 = typeof(op2)
    ELT = eltype(OP1,OP2)
    CompositeOperator2{OP1,OP2,ELT,length(size(src(op2))),SRC1,DEST2}(op1,op2)
end

src(op::CompositeOperator2) = src(op.op1)

dest(op::CompositeOperator2) = dest(op.op2)

eltype{OP1,OP2,ELT,N,SRC,DEST}(::Type{CompositeOperator2{OP1,OP2,ELT,N,SRC,DEST}}) = ELT

ctranspose(op::CompositeOperator2) = CompositeOperator2(ctranspose(op.op2), ctranspose(op.op1))

inv(op::CompositeOperator2) = CompositeOperator2(inv(op.op2), inv(op.op1))


apply!(op::CompositeOperator2, dest, src, coef_dest, coef_src) = _apply!(op, is_inplace(op.op2), coef_dest, coef_src)


function _apply!(op::CompositeOperator2, op2_inplace::True, coef_dest, coef_src)
    apply!(op.op1, coef_dest, coef_src)
    apply!(op.op2, coef_dest)
end

function _apply!(op::CompositeOperator2, op2_inplace::False, coef_dest, coef_src)
    apply!(op.op1, op.scratch, coef_src)
    apply!(op.op2, coef_dest, op.scratch)
end


# In-place operation: the problem is we can not simply assume that all operators are in-place, even if it is
# assured that the final result can be overwritten in coef_srcdest. Depending on the in-place-ness of the
# intermediate operators we have to resort to using scratch space.
apply!(op::CompositeOperator2, dest, src, coef_srcdest) =
    _apply_inplace!(op, is_inplace(op.op1), is_inplace(op.op2), coef_srcdest)

# In-place if all operators are in-place
function _apply_inplace!(op::CompositeOperator2, op1_inplace::True, op2_inplace::True, coef_srcdest)
    apply!(op.op1, coef_srcdest)
    apply!(op.op2, coef_srcdest)
end

# Is either one of the operator is not in-place, we have to use scratch space.
function _apply_inplace!(op::CompositeOperator2, op1_inplace, op2_inplace, coef_srcdest)
    apply!(op.op1, op.scratch, coef_srcdest)
    apply!(op.op2, coef_srcdest, op.scratch)
end


is_inplace{OP1,OP2,ELT,N,SRC,DEST}(::Type{CompositeOperator2{OP1,OP2,ELT,N,SRC,DEST}}) = is_inplace(OP1) & is_inplace(OP2)

is_diagonal{OP1,OP2,ELT,N,SRC,DEST}(::Type{CompositeOperator2{OP1,OP2,ELT,N,SRC,DEST}}) = is_diagonal(OP1) & is_diagonal(OP2)


"A composite operator with exactly three operators."
immutable CompositeOperator3{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST} <: AbstractOperator{SRC,DEST}
    op1         ::  OP1
    op2         ::  OP2
    op3         ::  OP3
    scratch1    ::  Array{ELT,N1}   # For storing the intermediate result after applying op1
    scratch2    ::  Array{ELT,N2}   # For storing the intermediate result after applying op2

    function CompositeOperator3(op1::AbstractOperator, op2::AbstractOperator, op3::AbstractOperator)
        @assert size(op1,1) == size(op2,2)
        @assert size(op2,1) == size(op3,2)

        new(op1, op2, op3, zeros(ELT,size(src(op2))), zeros(ELT,size(src(op3))))
    end
end

function CompositeOperator3{SRC1,DEST1,SRC3,DEST3}(op1::AbstractOperator{SRC1,DEST1}, op2, op3::AbstractOperator{SRC3,DEST3})
    OP1 = typeof(op1)
    OP2 = typeof(op2)
    OP3 = typeof(op3)
    ELT = eltype(OP1,OP2,OP3)
    N1 = length(size(src(op2)))
    N2 = length(size(src(op3)))
    CompositeOperator3{OP1,OP2,OP3,ELT,N1,N2,SRC1,DEST3}(op1, op2, op3)
end

src(op::CompositeOperator3) = src(op.op1)

dest(op::CompositeOperator3) = dest(op.op3)

eltype{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}(::Type{CompositeOperator3{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}}) = ELT

is_inplace{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}(::Type{CompositeOperator3{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}}) =
    is_inplace(OP1) & is_inplace(OP2) & is_inplace(OP3)

is_diagonal{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}(::Type{CompositeOperator3{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}}) =
    is_diagonal(OP1) & is_diagonal(OP2) & is_diagonal(OP3)

ctranspose(op::CompositeOperator3) = CompositeOperator3(ctranspose(op.op3), ctranspose(op.op2), ctranspose(op.op1))

inv(op::CompositeOperator3) = CompositeOperator3(inv(op.op3), inv(op.op2), inv(op.op1))

apply!(op::CompositeOperator3, dest, src, coef_dest, coef_src) =
    _apply!(op, is_inplace(op.op2), is_inplace(op.op3), coef_dest, coef_src)

function _apply!(op::CompositeOperator3, op2_inplace::True, op3_inplace::True, coef_dest, coef_src)
    apply!(op.op1, coef_dest, coef_src)
    apply!(op.op2, coef_dest)
    apply!(op.op3, coef_dest)
end

function _apply!(op::CompositeOperator3, op2_inplace::True, op3_inplace::False, coef_dest, coef_src)
    apply!(op.op1, op.scratch2, coef_src)
    apply!(op.op2, op.scratch2)
    apply!(op.op3, coef_dest, op.scratch2)
end

function _apply!(op::CompositeOperator3, op2_inplace::False, op3_inplace::True, coef_dest, coef_src)
    apply!(op.op1, op.scratch1, coef_src)
    apply!(op.op2, coef_dest, op.scratch1)
    apply!(op.op3, coef_dest)
end

function _apply!(op::CompositeOperator3, op2_inplace::False, op3_inplace::False, coef_dest, coef_src)
    apply!(op.op1, op.scratch1, coef_src)
    apply!(op.op2, op.scratch2, op.scratch1)
    apply!(op.op3, coef_dest, op.scratch2)
end


# In-place operation: the problem is we can not simply assume that all operators are in-place, even if it is
# assured that the final result can be overwritten in coef_srcdest. Depending on the in-place-ness of the
# intermediate operators we have to resort to using scratch space.
apply!(op::CompositeOperator3, dest, src, coef_srcdest) =
    _apply_inplace!(op, is_inplace(op.op1), is_inplace(op.op2), is_inplace(op.op3), coef_srcdest)

# If operator 1 is not in place, we can simply call the non-inplace version with coef_srcdest as src and dest.
_apply_inplace!(op::CompositeOperator3, op1_inplace::False, op2_inplace, op3_inplace, coef_srcdest) =
    _apply!(op, op2_inplace, op3_inplace, coef_srcdest, coef_srcdest)

# If operator 1 is in place, we have to do things ourselves.
# If either one of op2 or op3 is not in-place, we use scratch space
function _apply_inplace!(op::CompositeOperator3, op1_inplace::True, op2_inplace, op3_inplace, coef_srcdest)
    apply!(op.op1, coef_srcdest)
    apply!(op.op2, op.scratch2, coef_srcdest)
    apply!(op.op1, coef_srcdest, op.scratch2)
end

# We can avoid using scratch2 only when all operators are in-place
function _apply_inplace!(op::CompositeOperator3, op1_inplace::True, op2_inplace::True, op3_inplace::True, coef_srcdest)
    apply!(op.op1, coef_srcdest)
    apply!(op.op2, coef_srcdest)
    apply!(op.op3, coef_srcdest)
end
