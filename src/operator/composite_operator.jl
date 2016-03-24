# composite_operator.jl



"""
A composite operator applies op2 after op1.
It preallocates sufficient memory to store intermediate results.
"""
immutable CompositeOperator{OP1,OP2,ELT,N,SRC,DEST} <: AbstractOperator{SRC,DEST}
    op1     ::  OP1
    op2     ::  OP2
    scratch ::  Array{ELT,N}    # For storing the intermediate result after applying op1

    function CompositeOperator(op1::AbstractOperator, op2::AbstractOperator)
        @assert size(op1,1) == size(op2,2)

        # Possible optimization here would be to avoid allocating memory if the second operator is in-place.
        # But even in that case, the user may invoke the operator in a non-in-place way, so let's keep it.
        new(op1, op2, zeros(ELT,size(src(op2))))
    end
end


# We could ask that DEST1 == SRC2 but that might be too strict. As long as the operators are compatible things are fine.
function CompositeOperator{SRC1,DEST1,SRC2,DEST2}(op1::AbstractOperator{SRC1,DEST1}, op2::AbstractOperator{SRC2,DEST2})
        #@assert DEST1 == SRC2
    OP1 = typeof(op1)
    OP2 = typeof(op2)
    ELT = eltype(OP1,OP2)
    CompositeOperator{OP1,OP2,ELT,length(size(src(op2))),SRC1,DEST2}(op1,op2)
end

src(op::CompositeOperator) = src(op.op1)

dest(op::CompositeOperator) = dest(op.op2)

eltype{OP1,OP2,ELT,N,SRC,DEST}(::Type{CompositeOperator{OP1,OP2,ELT,N,SRC,DEST}}) = ELT

ctranspose(op::CompositeOperator) = CompositeOperator(ctranspose(op.op2), ctranspose(op.op1))

inv(op::CompositeOperator) = CompositeOperator(inv(op.op2), inv(op.op1))

(*)(op2::AbstractOperator, op1::AbstractOperator) = CompositeOperator(op1, op2)

apply!(op::CompositeOperator, dest, src, coef_dest, coef_src) = _apply!(op, is_inplace(op.op2), coef_dest, coef_src)


function _apply!(op::CompositeOperator, op2_inplace::True, coef_dest, coef_src)
    apply!(op.op1, coef_dest, coef_src)
    apply!(op.op2, coef_dest)
end

function _apply!(op::CompositeOperator, op2_inplace::False, coef_dest, coef_src)
    apply!(op.op1, op.scratch, coef_src)
    apply!(op.op2, coef_dest, op.scratch)
end


# In-place operation: the problem is we can not simply assume that all operators are in-place, even if it is
# assured that the final result can be overwritten in coef_srcdest. Depending on the in-place-ness of the 
# intermediate operators we have to resort to using scratch space.
apply!(op::CompositeOperator, dest, src, coef_srcdest) =
    _apply_inplace!(op, is_inplace(op.op1), is_inplace(op.op2), coef_srcdest)

# In-place if all operators are in-place
function _apply_inplace!(op::CompositeOperator, op1_inplace::True, op2_inplace::True, coef_srcdest)
    apply!(op.op1, coef_srcdest)
    apply!(op.op2, coef_srcdest)
end

# Is either one of the operator is not in-place, we have to use scratch space.
function _apply_inplace!(op::CompositeOperator, op1_inplace, op2_inplace, coef_srcdest)
    apply!(op.op1, op.scratch, coef_srcdest)
    apply!(op.op2, coef_srcdest, op.scratch)
end


is_inplace{OP1,OP2,ELT,N,SRC,DEST}(::Type{CompositeOperator{OP1,OP2,ELT,N,SRC,DEST}}) = is_inplace(OP1) & is_inplace(OP2)

is_diagonal{OP1,OP2,ELT,N,SRC,DEST}(::Type{CompositeOperator{OP1,OP2,ELT,N,SRC,DEST}}) = is_diagonal(OP1) & is_diagonal(OP2)


# Perhaps the parameters are excessive in this and many other types in the code. Do some tests without them later.
# A TripleCompositeOperator is implemented because it can better exploit in-place operators
# than a chain of CompositeOperator's.
immutable TripleCompositeOperator{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST} <: AbstractOperator{SRC,DEST}
    op1         ::  OP1
    op2         ::  OP2
    op3         ::  OP3
    scratch1    ::  Array{ELT,N1}   # For storing the intermediate result after applying op1
    scratch2    ::  Array{ELT,N2}   # For storing the intermediate result after applying op2

    function TripleCompositeOperator(op1::AbstractOperator, op2::AbstractOperator, op3::AbstractOperator)
        @assert size(op1,1) == size(op2,2)
        @assert size(op2,1) == size(op3,2)

        new(op1, op2, op3, zeros(ELT,size(src(op2))), zeros(ELT,size(src(op3))))
    end
end

function TripleCompositeOperator{SRC1,DEST1,SRC3,DEST3}(op1::AbstractOperator{SRC1,DEST1}, op2, op3::AbstractOperator{SRC3,DEST3})
    OP1 = typeof(op1)
    OP2 = typeof(op2)
    OP3 = typeof(op3)
    ELT = eltype(OP1,OP2,OP3)
    N1 = length(size(src(op2)))
    N2 = length(size(src(op3)))
    TripleCompositeOperator{OP1,OP2,OP3,ELT,N1,N2,SRC1,DEST3}(op1, op2, op3)
end

src(op::TripleCompositeOperator) = src(op.op1)

dest(op::TripleCompositeOperator) = dest(op.op3)

eltype{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}(::Type{TripleCompositeOperator{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}}) = ELT

is_inplace{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}(::Type{TripleCompositeOperator{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}}) =
    is_inplace(OP1) & is_inplace(OP2) & is_inplace(OP3)

is_diagonal{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}(::Type{TripleCompositeOperator{OP1,OP2,OP3,ELT,N1,N2,SRC,DEST}}) =
    is_diagonal(OP1) & is_diagonal(OP2) & is_diagonal(OP3)

ctranspose(op::TripleCompositeOperator) = TripleCompositeOperator(ctranspose(op.op3), ctranspose(op.op2), ctranspose(op.op1))

inv(op::TripleCompositeOperator) = TripleCompositeOperator(inv(op.op3), inv(op.op2), inv(op.op1))

apply!(op::TripleCompositeOperator, dest, src, coef_dest, coef_src) =
    _apply!(op, is_inplace(op.op2), is_inplace(op.op3), coef_dest, coef_src)

function _apply!(op::TripleCompositeOperator, op2_inplace::True, op3_inplace::True, coef_dest, coef_src)
    apply!(op.op1, coef_dest, coef_src)
    apply!(op.op2, coef_dest)
    apply!(op.op3, coef_dest)
end

function _apply!(op::TripleCompositeOperator, op2_inplace::True, op3_inplace::False, coef_dest, coef_src)
    apply!(op.op1, op.scratch2, coef_src)
    apply!(op.op2, op.scratch2)
    apply!(op.op3, coef_dest, op.scratch2)
end

function _apply!(op::TripleCompositeOperator, op2_inplace::False, op3_inplace::True, coef_dest, coef_src)
    apply!(op.op1, op.scratch1, coef_src)
    apply!(op.op2, coef_dest, op.scratch1)
    apply!(op.op3, coef_dest)
end

function _apply!(op::TripleCompositeOperator, op2_inplace::False, op3_inplace::False, coef_dest, coef_src)
    apply!(op.op1, op.scratch1, coef_src)
    apply!(op.op2, op.scratch2, op.scratch1)
    apply!(op.op3, coef_dest, op.scratch2)
end


# In-place operation: the problem is we can not simply assume that all operators are in-place, even if it is
# assured that the final result can be overwritten in coef_srcdest. Depending on the in-place-ness of the 
# intermediate operators we have to resort to using scratch space.
apply!(op::TripleCompositeOperator, dest, src, coef_srcdest) =
    _apply_inplace!(op, is_inplace(op.op1), is_inplace(op.op2), is_inplace(op.op3), coef_srcdest)

# If operator 1 is not in place, we can simply call the non-inplace version with coef_srcdest as src and dest.
_apply_inplace!(op::TripleCompositeOperator, op1_inplace::False, op2_inplace, op3_inplace, coef_srcdest) =
    _apply!(op, op2_inplace, op3_inplace, coef_srcdest, coef_srcdest)

# If operator 1 is in place, we have to do things ourselves.
# If either one of op2 or op3 is not in-place, we use scratch space
function _apply_inplace!(op::TripleCompositeOperator, op1_inplace::True, op2_inplace, op3_inplace, coef_srcdest)
    apply!(op.op1, coef_srcdest)
    apply!(op.op2, op.scratch2, coef_srcdest)
    apply!(op.op1, coef_srcdest, op.scratch2)
end

# We can avoid using scratch2 only when all operators are in-place
function _apply_inplace!(op::TripleCompositeOperator, op1_inplace::True, op2_inplace::True, op3_inplace::True, coef_srcdest)
    apply!(op.op1, coef_srcdest)
    apply!(op.op2, coef_srcdest)
    apply!(op.op3, coef_srcdest)
end


(*)(op3::AbstractOperator, op2::AbstractOperator, op1::AbstractOperator) = TripleCompositeOperator(op1, op2, op3)


