# composite_operator.jl


"""
A `GenericCompositeOperator` contains a list of operators that are applied
consecutively to any input. It is a generic operator.
"""
struct GenericCompositeOperator <: AbstractOperator
    operators
end

# This constructor takes several operators as arguments and ensures that
# the spaces are compatible. This step can by sidestepped by calling the
# default inner constructor directly with a vector or tuple of operators instead.
function GenericCompositeOperator(operators::AbstractOperator...)
    for i in 1:length(operators)-1
        @assert coeftype(dest_space(operators[i])) == coeftype(src_space(operators[i+1]))
    end
    # Pass the tuple of operators to the inner constructor
    GenericCompositeOperator(operators)
end

src_space(op::GenericCompositeOperator) = src_space(op.operators[1])
dest_space(op::GenericCompositeOperator) = dest_space(op.operators[end])

# Generic functions for composite types:
elements(op::GenericCompositeOperator) = op.operators
element(op::GenericCompositeOperator, j::Int) = op.operators[j]
is_composite(op::GenericCompositeOperator) = true

function apply(comp::GenericCompositeOperator, fun)
    output = fun
    for op in elements(comp)
        input = output
        output = apply(op, input)
    end
    output
end

"Can an operator allocate storage for its expected output?"
# The answer is no in general, but yes if the output is the span of a dictionary
# - Yes for all dictionary operators
can_allocate_output(op::DictionaryOperator) = true
# - AbstractOperator: it depends on the output, which we determine using dispatch
can_allocate_output(op::AbstractOperator) = _can_allocate_output(op, dest_space(op))
_can_allocate_output(op, span::Span) = true
_can_allocate_output(op, space::AbstractFunctionSpace) = false

allocate_output(op::AbstractOperator) = zeros(dest(op))

"""
A `CompositeOperator` consists of a sequence of operators that are applied
consecutively.

Whenever possible, scratch space is allocated to hold intermediate results.
"""
struct CompositeOperator{T} <: DictionaryOperator{T}
    # We explicitly store src and dest, because that information may be lost
    # when the list of operators is optimized (for example, an Identity mapping
    # between two spaces could disappear).
    src     ::  Dictionary
    dest    ::  Dictionary
    "The list of operators"
    operators
    "Scratch space for the result of each operator, except the last one"
    # We don't need it for the last one, because the final result goes to coef_dest
    scratch
end

# Generic functions for composite types:
elements(op::CompositeOperator) = op.operators
element(op::CompositeOperator, j::Int) = op.operators[j]

is_inplace(op::CompositeOperator) = reduce(&, map(is_inplace, op.operators))
is_diagonal(op::CompositeOperator) = reduce(&, map(is_diagonal, op.operators))
is_composite(op::CompositeOperator) = true


CompositeOperator(operators::DictionaryOperator...) =
    CompositeOperator(src(operators[1]), dest(operators[end]), operators...)

function CompositeOperator(composite_src::Dictionary, composite_dest::Dictionary, operators::DictionaryOperator...)
    L = length(operators)
    # Check operator compatibility
    for i in 1:length(operators)-1
        @assert size(dest(operators[i])) == size(src(operators[i+1]))
    end

    # TODO: Not sure if this is the right thing to do here
    T = promote_type(map(eltype, operators)...,coeftype(composite_src),coeftype(composite_dest))
    c_operators = map(o -> promote_eltype(o, T), operators)
    # We are going to reserve scratch space, but only for operators that are not
    # in-place. We do reserve scratch space for the first operator, even if it
    # is in-place, because we may want to call the composite operator out of place.
    # In that case we need a place to store the result of the first operator.
    scratch_array = Any[zeros(dest(c_operators[1]))]
    for m = 2:L-1
        if ~is_inplace(c_operators[m])
            push!(scratch_array, zeros(dest(c_operators[m])))
        end
    end
    scratch = tuple(scratch_array...)
    CompositeOperator{T}(composite_src, composite_dest, c_operators, scratch)
end

similar_operator(op::CompositeOperator, ::Type{S}, src, dest) where {S} =
    CompositeOperator(promote_coeftype(src, S), promote_coeftype(dest, S), map(o-> promote_eltype(o,S),elements(op))...)

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
(∘)(ops::AbstractOperator...) = (*)(ops...)
apply(op1::AbstractOperator, op2::AbstractOperator) = compose(op2,op1)
apply(op1::DictionaryOperator, op2::AbstractOperator) = compose(op2,op1)

# Don't do anything if we have just one operator
compose(op::AbstractOperator) = op

# Here we have at least two operators. Remove nested compositions with flatten and continue.
compose(ops::DictionaryOperator...) = CompositeOperator(flatten(CompositeOperator, ops...)...)
compose(ops::AbstractOperator...) = GenericCompositeOperator(flatten(GenericCompositeOperator, ops...)...)

sparse_matrix(op::CompositeOperator; options...) = *([sparse_matrix(opi; options...) for opi in elements(op)[end:-1:1]]...)

CompositeOperators = Union{CompositeOperator,GenericCompositeOperator}

function stencil(op::CompositeOperators)
    A = Any[]
    push!(A,element(op,length(elements(op))))
    for i=length(elements(op))-1:-1:1
        push!(A," * ")
        push!(A,element(op,i))
    end
    A
end
