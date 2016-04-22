# composite_operator.jl

"""
A composite operator consists of a sequence of operators that are applied
consecutively.
"""
immutable CompositeOperator{ELT} <: AbstractOperator{ELT}
    "The list of operators"
    operators
    "Scratch space for the result of each operator, except the last one"
    # We don't need it for the last one, because the final result goes to coef_dest
    scratch
end

# Generic functions for composite types:
elements(op::CompositeOperator) = op.operators
element(op::CompositeOperator, j::Int) = op.operators[j]
composite_length(op::CompositeOperator) = length(op.operators)

src(op::CompositeOperator, j::Int = 1) = src(op.operators[j])
dest(op::CompositeOperator, j::Int = composite_length(op)) = dest(op.operators[j])

is_inplace(op::CompositeOperator) = reduce(&, map(is_inplace, op.operators))
is_diagonal(op::CompositeOperator) = reduce(&, map(is_diagonal, op.operators))


function CompositeOperator(operators::AbstractOperator...)
    L = length(operators)
    # Check operator compatibility
    for i in 1:length(operators)-1
        @assert size(dest(operators[i])) == size(src(operators[i+1]))
    end

    ELT = promote_type(map(eltype, operators)...)
    # We are going to reserve scratch space, but only for operators that are not
    # in-place. We do reserve scratch space for the first operator, even if it
    # is in-place, because we may want to call the composite operator out of place.
    # In that case we need a place to store the result of the first operator.
    scratch_array = Any[zeros(ELT, size(dest(operators[1])))]
    for m = 2:L-1
        if ~is_inplace(operators[m])
            push!(scratch_array, zeros(ELT, size(dest(operators[m]))))
        end
    end
    scratch = tuple(scratch_array...)
    # CompositeOperator{ELT}([operators...], scratch_array)
    CompositeOperator{ELT}(operators, scratch)
end

apply_inplace!(op::CompositeOperator, coef_srcdest) =
    apply_inplace_composite!(op, coef_srcdest, op.operators)

apply!(op::CompositeOperator, coef_dest, coef_src) =
    apply_composite!(op, coef_dest, coef_src, op.operators, op.scratch)


function apply_composite!(op::CompositeOperator, coef_dest, coef_src, operators, scratch)
    L = length(operators)
    apply!(operators[1], scratch[1], coef_src)
    l = 1
    for i in 2:L-1
        if is_inplace(operators[i])
            apply_inplace!(operators[i], scratch[l])
        else
            apply!(operators[i], scratch[l+1], scratch[l])
            l += 1
        end
    end
    # We loose a little bit of efficiency if the last operator was in-place
    apply!(operators[L], coef_dest, scratch[l])
    # Return coef_dest even though the last apply! should also do that, to
    # help type-inference continue afterwards
    coef_dest
end

# Below is the ideal scenario for lengths 2 and 3, written explicitly.
# function apply_composite!(op::CompositeOperator, coef_dest, coef_src, operators::NTuple{2}, scratch)
#     if is_inplace(operators[2])
#         apply!(operators[1], coef_dest, coef_src)
#         apply!(operators[2], coef_dest)
#     else
#         apply!(operators[1], scratch[1], coef_src)
#         apply!(operators[2], coef_dest, scratch[1])
#     end
#     coef_dest
# end
#
# function apply_composite!(op::CompositeOperator, coef_dest, coef_src, operators::NTuple{3}, scratch)
#     ip2 = is_inplace(operators[2])
#     ip3 = is_inplace(operators[3])
#     if ip2
#         if ip3
#             # 2 and 3 are in-place
#             apply!(operators[1], coef_dest, coef_src)
#             apply!(operators[2], coef_dest)
#             apply!(operators[3], coef_dest)
#         else
#             # 2 is in place, 3 is not
#             apply!(operators[1], scratch[1], coef_src)
#             apply!(operators[2], scratch[1])
#             apply!(operators[3], coef_dest, scratch[1])
#         end
#     else
#         if ip3
#             # 2 is not in place, but 3 is
#             apply!(operators[1], scratch[1], coef_src)
#             apply!(operators[2], coef_dest, scratch[1])
#             apply!(operators[3], coef_dest)
#         else
#             # noone is in place
#             apply!(operators[1], scratch[1], coef_src)
#             apply!(operators[2], scratch[2], scratch[1])
#             apply!(operators[3], coef_dest, scratch[2])
#         end
#     end
#     coef_dest
# end

function apply_inplace_composite!(op::CompositeOperator, coef_srcdest, operators)
    for operator in operators
        apply!(operator, coef_srcdest)
    end
    coef_srcdest
end

inv(op::CompositeOperator) = (*)(map(inv, op.operators)...)

ctranspose(op::CompositeOperator) = (*)(map(ctranspose, op.operators)...)

compose() = nothing
compose(ops::AbstractOperator...) = CompositeOperator(flatten(CompositeOperator, ops...)...)

(*)(ops::AbstractOperator...) = compose([ops[i] for i in length(ops):-1:1]...)


# This is the old way, which takes way more compilation time because everything
# is typed and composition of many operators is nested:
# (*)(op2::AbstractOperator, op1::AbstractOperator) = CompositeOperator2(op1, op2)
# (*)(op3::AbstractOperator, op2::AbstractOperator, op1::AbstractOperator) = CompositeOperator3(op1, op2, op3)


"""
A composite operator of exactly two operators.
"""
immutable CompositeOperator2{OP1,OP2,ELT,N} <: AbstractOperator{ELT}
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
function CompositeOperator2(op1::AbstractOperator, op2::AbstractOperator)
        #@assert DEST1 == SRC2
    OP1 = typeof(op1)
    OP2 = typeof(op2)
    ELT = promote_type(eltype(op1), eltype(op2))
    N = index_dim(src(op2))
    CompositeOperator2{OP1,OP2,ELT,N}(op1,op2)
end

src(op::CompositeOperator2) = src(op.op1)

dest(op::CompositeOperator2) = dest(op.op2)

ctranspose(op::CompositeOperator2) = CompositeOperator2(ctranspose(op.op2), ctranspose(op.op1))

inv(op::CompositeOperator2) = CompositeOperator2(inv(op.op2), inv(op.op1))

is_inplace(op::CompositeOperator2) = is_inplace(op.op1) & is_inplace(op.op2)
is_diagonal(op::CompositeOperator2) = is_diagonal(op.op1) & is_diagonal(op.op2)


apply!(op::CompositeOperator2, coef_dest, coef_src) = _apply!(op, is_inplace(op.op2), coef_dest, coef_src)


function _apply!(op::CompositeOperator2, op2_inplace::Bool, coef_dest, coef_src)
    if op2_inplace
        apply!(op.op1, coef_dest, coef_src)
        apply!(op.op2, coef_dest)
    else
        apply!(op.op1, op.scratch, coef_src)
        apply!(op.op2, coef_dest, op.scratch)
    end
    coef_dest
end


# In-place operation: the problem is we can not simply assume that all operators are in-place, even if it is
# assured that the final result can be overwritten in coef_srcdest. Depending on the in-place-ness of the
# intermediate operators we have to resort to using scratch space.
apply_inplace!(op::CompositeOperator2, coef_srcdest) =
    _apply_inplace!(op, is_inplace(op.op1), is_inplace(op.op2), coef_srcdest)

# In-place if all operators are in-place
function _apply_inplace!(op::CompositeOperator2, op1_inplace::Bool, op2_inplace::Bool, coef_srcdest)
    if op1_inplace && op2_inplace
        apply!(op.op1, coef_srcdest)
        apply!(op.op2, coef_srcdest)
    else
        apply!(op.op1, op.scratch, coef_srcdest)
        apply!(op.op2, coef_srcdest, op.scratch)
    end
    coef_srcdest
end


"A composite operator with exactly three operators."
immutable CompositeOperator3{OP1,OP2,OP3,ELT,N1,N2} <: AbstractOperator{ELT}
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

function CompositeOperator3(op1::AbstractOperator, op2::AbstractOperator, op3::AbstractOperator)
    OP1 = typeof(op1)
    OP2 = typeof(op2)
    OP3 = typeof(op3)
    ELT = eltype(OP1,OP2,OP3)
    N1 = length(size(src(op2)))
    N2 = length(size(src(op3)))
    CompositeOperator3{OP1,OP2,OP3,ELT,N1,N2}(op1, op2, op3)
end

src(op::CompositeOperator3) = src(op.op1)

dest(op::CompositeOperator3) = dest(op.op3)


is_inplace(op::CompositeOperator3) = is_inplace(op.op1) & is_inplace(op.op2) & is_inplace(op.op3)
is_diagonal(op::CompositeOperator3) = is_diagonal(op.op1) & is_diagonal(op.op2) & is_diagonal(op.op3)

ctranspose(op::CompositeOperator3) = CompositeOperator3(ctranspose(op.op3), ctranspose(op.op2), ctranspose(op.op1))

inv(op::CompositeOperator3) = CompositeOperator3(inv(op.op3), inv(op.op2), inv(op.op1))

apply!(op::CompositeOperator3, coef_dest, coef_src) =
    _apply!(op, is_inplace(op.op2), is_inplace(op.op3), coef_dest, coef_src)

function _apply!(op::CompositeOperator3, op2_inplace::Bool, op3_inplace::Bool, coef_dest, coef_src)
    if op2_inplace && op3_inplace
        apply!(op.op1, coef_dest, coef_src)
        apply_inplace!(op.op2, coef_dest)
        apply_inplace!(op.op3, coef_dest)
    elseif op2_inplace && ~op3_inplace
        apply!(op.op1, op.scratch2, coef_src)
        apply_inplace!(op.op2, op.scratch2)
        apply!(op.op3, coef_dest, op.scratch2)
    elseif ~op2_inplace && op3_inplace
        apply!(op.op1, op.scratch1, coef_src)
        apply!(op.op2, coef_dest, op.scratch1)
        apply_inplace!(op.op3, coef_dest)
    else
        apply!(op.op1, op.scratch1, coef_src)
        apply!(op.op2, op.scratch2, op.scratch1)
        apply!(op.op3, coef_dest, op.scratch2)
    end
    coef_dest
end


function apply_inplace!(op::CompositeOperator3, coef_srcdest)
    apply!(op.op1, coef_srcdest)
    apply!(op.op2, coef_srcdest)
    apply!(op.op3, coef_srcdest)
end

# # In-place operation: the problem is we can not simply assume that all operators are in-place, even if it is
# # assured that the final result can be overwritten in coef_srcdest. Depending on the in-place-ness of the
# # intermediate operators we have to resort to using scratch space.
# apply_inplace!(op::CompositeOperator3, coef_srcdest) =
#     _apply_inplace!(op, is_inplace(op.op1), is_inplace(op.op2), is_inplace(op.op3), coef_srcdest)
#
# # If operator 1 is not in place, we can simply call the non-inplace version with coef_srcdest as src and dest.
# _apply_inplace!(op::CompositeOperator3, op1_inplace::False, op2_inplace, op3_inplace, coef_srcdest) =
#     _apply!(op, op2_inplace, op3_inplace, coef_srcdest, coef_srcdest)
#
# # If operator 1 is in place, we have to do things ourselves.
# # If either one of op2 or op3 is not in-place, we use scratch space
# function _apply_inplace!(op::CompositeOperator3, op1_inplace::True, op2_inplace, op3_inplace, coef_srcdest)
#     apply!(op.op1, coef_srcdest)
#     apply!(op.op2, op.scratch2, coef_srcdest)
#     apply!(op.op1, coef_srcdest, op.scratch2)
# end
#
# # We can avoid using scratch2 only when all operators are in-place
# function _apply_inplace!(op::CompositeOperator3, op1_inplace::True, op2_inplace::True, op3_inplace::True, coef_srcdest)
#     apply!(op.op1, coef_srcdest)
#     apply!(op.op2, coef_srcdest)
#     apply!(op.op3, coef_srcdest)
# end
