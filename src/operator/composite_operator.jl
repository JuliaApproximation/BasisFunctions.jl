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
