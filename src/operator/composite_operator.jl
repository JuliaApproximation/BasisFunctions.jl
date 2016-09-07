# composite_operator.jl

"""
A CompositeOperator consists of a sequence of operators that are applied
consecutively.
Memory is allocated at creation time to hold intermediate results.
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
    scratch_array = Any[zeros(ELT, dest(operators[1]))]
    for m = 2:L-1
        if ~is_inplace(operators[m])
            push!(scratch_array, zeros(ELT, dest(operators[m])))
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

(*)(ops::AbstractOperator...) = compose([ops[i] for i in length(ops):-1:1]...)

# Don't do anything if we have just one operator
compose(op::AbstractOperator) = op

# Here we have at least two operators. Remove nested compositions with flatten and continue.
compose(ops::AbstractOperator...) = compose_verify_and_simplify(flatten(CompositeOperator, ops...)...)
# compose(ops::AbstractOperator...) = CompositeOperator(flatten(CompositeOperator, ops...)...)

function compose_verify_and_simplify(ops::AbstractOperator...)
    # Check for correct chain of function spaces
    for i in 1:length(ops)-1
        dest(ops[i]) == src(ops[i+1]) || error("Size and destination don't match in ", typeof(ops[i]), " and ", typeof(ops[i+1]))
    end
    # Initiate recursive simplification
    compose_simplify_rec( src(ops[1]), dest(ops[end]), [], ops[1], ops[2:end]...)
end

# We attempt to simplify the composition of operators with the following rules:
# - each operator is first simplified on its own (e.g. WrappedOperator can remove the wrap,
#   IdentityOperator can disappear)
# - Each operator is then compared with the next one, so that pairs of operators can be simplified

# The function compose_simplify_rec expects the overall source and destination of the
# chain, an array prev that has already been processed, and the remaining operators as
# individual arguments.

# Only prev is specified: we have processed all operators and we are done
compose_simplify_rec(src, dest, prev) = compose_simplify_done(src, dest, prev...)

# One extra argument: we have one operator left to examine
function compose_simplify_rec(src, dest, prev, current)
    simple_current = simplify(current)
    if simple_current == nothing
        compose_simplify_done(src, dest, prev...)
    else
        compose_simplify_done(src, dest, prev..., simple_current)
    end
end

# There is a next operator and zero or more remaining operators
function compose_simplify_rec(src, dest, prev, current, next, remaining::AbstractOperator...)
    simple_current = simplify(current)
    if simple_current == nothing
        compose_simplify_rec(src, dest, prev, next, remaining...)
    else
        simple_pair = simplify(simple_current, next)
        if length(simple_pair) == 0
            compose_simplify_rec(src, dest, prev, remaining...)
        elseif length(simple_pair) == 1
            compose_simplify_rec(src, dest, prev, simple_pair[1], remaining...)
        else
            compose_simplify_rec(src, dest, [prev; simple_pair[1:end-1]...], simple_pair[end], remaining...)
        end
    end
end

# By default, simplification does nothing
simplify(op::AbstractOperator) = op
simplify(op1::AbstractOperator, op2::AbstractOperator) = (op1,op2)

# When nothing remains, construct an identity operator from src to dest.
# This is the reason for passing around those arguments all the time.
compose_simplify_done(src, dest) = IdentityOperator(src, dest)

# Do nothing for a single operator
compose_simplify_done(src, dest, op::AbstractOperator) = op

# Construct a composite operator
compose_simplify_done(src, dest, op1::AbstractOperator, op2::AbstractOperator, ops::AbstractOperator...) = CompositeOperator(op1, op2, ops...)
