
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
        @assert coefficienttype(dest_space(operators[i])) == coefficienttype(src_space(operators[i+1]))
    end
    # Pass the tuple of operators to the inner constructor
    GenericCompositeOperator(operators)
end

src_space(op::GenericCompositeOperator) = src_space(op.operators[1])
dest_space(op::GenericCompositeOperator) = dest_space(op.operators[end])

# Generic functions for composite types:
elements(op::GenericCompositeOperator) = op.operators
element(op::GenericCompositeOperator, j::Int) = op.operators[j]
iscomposite(op::GenericCompositeOperator) = true

# If the GenericCompositeOperator happens to be a composite operator of
# product operators, then the result is a product operator too. You can select
# its elements with the productelement routine.
function productelement(op::GenericCompositeOperator, j::Int)
    GenericCompositeOperator(map(t -> element(t, j), elements(op))...)
end
productelements(op::GenericCompositeOperator) = tuple([productelement(op, j) for j in 1:numproductelements(op)]...)
numproductelements(op::GenericCompositeOperator) = numproductelements(element(op,1))

function apply(comp::GenericCompositeOperator, fun; options...)
    output = fun
    for op in elements(comp)
        input = output
        # TODO: clean this op. The apply of a generic operator can accept options,
        # but the apply of a DictionaryOperator can not.
        if op isa DictionaryOperator
            output = apply(op, input)
        else
            output = apply(op, input; options...)
        end
    end
    output
end

"Can an operator allocate storage for its expected output?"
can_allocate_output(op::DictionaryOperator) = true
# The answer is no in general, but yes if the output is the span of a dictionary
# - Yes for all dictionary operators

# - AbstractOperator: it depends on the output, which we determine using dispatch
can_allocate_output(op::AbstractOperator) = _can_allocate_output(op, dest_space(op))
_can_allocate_output(op, span::Span) = true
_can_allocate_output(op, space::FunctionSpace) = false

allocate_output(op::DictionaryOperator{T}) where {T} = zeros(T, dest(op))

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

function productelement(op::CompositeOperator, j::Int)
    compose(map(t -> element(t, j), elements(op))...)
end
productelements(op::CompositeOperator) = tuple([productelement(op, j) for j in 1:numproductelements(op)]...)
numproductelements(op::CompositeOperator) = numproductelements(element(op,1))

isinplace(op::CompositeOperator) = reduce(&, map(isinplace, op.operators))
isdiag(op::CompositeOperator) = reduce(&, map(isdiag, op.operators))
iscomposite(op::CompositeOperator) = true


function compose_and_simplify(composite_src::Dictionary, composite_dest::Dictionary, operators::DictionaryOperator...; simplify = true)
    # Check operator compatibility
    for i in 1:length(operators)-1
        @assert size(dest(operators[i])) == size(src(operators[i+1]))
#       TODO: at one point we should enable strict checking again as follows:
#        @assert dest(operators[i]) == src(operators[i+1])
    end

    # Checks pass, now apply some simplifications
    if simplify
        # Flatten away nested CompositeOperators
        operators = flatten(CompositeOperator, operators...)
        # Iterate over the operators and remove the ones that don't do anything
        c_operators = Array{AbstractOperator}(undef, 0)
        for op in operators
            add_thisone = true
            # We forget about identity operators
            if isa(op, IdentityOperator)
                add_thisone = false
            end
            if isa(op, ScalingOperator) && scalar(op) ≈ 1
                add_thisone = false
            end
            if add_thisone
                push!(c_operators, op)
            end
        end
        operators = tuple(c_operators...)
    end

    L = length(operators)
    if L == 0
        return IdentityOperator(composite_src, composite_dest)
    end
    if L == 1
        return wrap_operator(composite_src, composite_dest, operators[1])
    end
    T = promote_type(map(eltype, operators)...)
    # We are going to reserve scratch space, but only for operators that are not
    # in-place. We do reserve scratch space for the first operator, even if it
    # is in-place, because we may want to call the composite operator out of place.
    # In that case we need a place to store the result of the first operator.
    scratch_array = Any[zeros(T, dest(operators[1]))]
    for m = 2:L-1
        if ~isinplace(operators[m])
            push!(scratch_array, zeros(T, dest(operators[m])))
        end
    end
    scratch = tuple(scratch_array...)
    CompositeOperator{T}(composite_src, composite_dest, operators, scratch)
end

unsafe_wrap_operator(src, dest, op::CompositeOperator{T}) where T =
    CompositeOperator{T}(src, dest, op.operators, op.scratch)

apply_inplace!(op::CompositeOperator, coef_srcdest) =
    apply_inplace_composite!(op, coef_srcdest, op.operators)

apply!(op::CompositeOperator, coef_dest, coef_src) =
    apply_composite!(op, coef_dest, coef_src, op.operators, op.scratch)


function apply_composite!(op::CompositeOperator, coef_dest, coef_src, operators, scratch)
    L = length(operators)
    apply!(operators[1], scratch[1], coef_src)
    l = 1
    for i in 2:L-1
        if isinplace(operators[i])
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
#     if isinplace(operators[2])
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
#     ip2 = isinplace(operators[2])
#     ip3 = isinplace(operators[3])
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

adjoint(op::CompositeOperator)::DictionaryOperator = (*)(map(adjoint, op.operators)...)

apply(op1::AbstractOperator, op2::AbstractOperator) = compose(op2,op1)
apply(op1::DictionaryOperator, op2::AbstractOperator) = compose(op2,op1)

# Don't do anything if we have just one operator
compose(op::AbstractOperator) = op

compose(ops::DictionaryOperator...) = compose_and_simplify(src(ops[1]), dest(ops[end]), ops...)
compose(ops::AbstractOperator...) = GenericCompositeOperator(flatten(GenericCompositeOperator, ops...)...)

sparse_matrix(op::CompositeOperator; options...) = *([sparse_matrix(opi; options...) for opi in elements(op)[end:-1:1]]...)

CompositeOperators = Union{CompositeOperator,GenericCompositeOperator}

function stencilarray(op::CompositeOperators)
    A = Any[]
    push!(A,element(op,numelements(op)))
    for i in numelements(op)-1:-1:1
        push!(A, " * ")
        push!(A, element(op,i))
    end
    A
end

stencil_parentheses(op::CompositeOperators) = true
object_parentheses(op::CompositeOperators) = true
