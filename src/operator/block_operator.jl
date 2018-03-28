# block_operator.jl

"""
A BlockOperator has a block matrix structure, where each block is an operator,
and it acts on multiple sets.

A BlockOperator is row-like if it only has one row of blocks. In that case, the
destination of the operator is not necessarily a multidict.

A BlockOperator is column-like if it only has one column of blocks. In that case,
the source set of the operator is not necessarily a multidict.
"""
struct BlockOperator{T} <: ParentOperator{T}
    operators   ::  Array{AbstractOperator{T}, 2}
    src         ::  Span
    dest        ::  Span

    # scratch_src and scratch_dest hold scratch memory for each subset of the
    # source and destination sets, for allocation-free implementation of the
    # action of the operator
    scratch_src
    scratch_dest

    function BlockOperator{T}(operators, src, dest) where {T}
        scratch_src = zeros(src)
        scratch_dest = zeros(dest)
        new(operators, src, dest, scratch_src, scratch_dest)
    end
end

function BlockOperator(operators::Array{OP,2},
    op_src = multispan(map(src, operators[1,:])),
    op_dest = multispan(map(dest, operators[:,1]))) where {OP <: AbstractOperator}
    # Avoid 1x1 block operators
    @assert size(operators,1) + size(operators,2) > 2

    T = op_eltype(op_src, op_dest)
    BlockOperator{T}(operators, op_src, op_dest)
end

# sets... may contain src and dest sets, that will be passed on to the BlockOperator constructor
function block_row_operator(op1::AbstractOperator, op2::AbstractOperator, sets::Span...)
    T = promote_type(eltype(op1), eltype(op2))
    operators = Array{AbstractOperator{T}}(1, 2)
    operators[1] = op1
    operators[2] = op2
    BlockOperator(operators, sets...)
end

function block_row_operator(ops::Array{OP, 1}, sets::Span...) where {OP <: AbstractOperator}
    T = promote_type(map(eltype, ops)...)
    operators = Array{AbstractOperator{T}}(1, length(ops))
    operators[:] = ops
    BlockOperator(operators, sets...)
end

function block_column_operator(op1::AbstractOperator, op2::AbstractOperator)
    T = promote_type(eltype(op1), eltype(op2))
    operators = Array{AbstractOperator{T}}(2, 1)
    operators[1] = op1
    operators[2] = op2
    BlockOperator(operators)
end

function block_column_operator(ops::Array{OP, 1}) where {OP <: AbstractOperator}
    T = promote_type(map(eltype, ops)...)
    operators = Array{AbstractOperator{T}}(length(ops), 1)
    operators[:] = ops
    BlockOperator(operators)
end

# Return a block operator the size of the given operator, but filled with zero
# operators.
function zeros(op::BlockOperator)
    operators = zeros(AbstractOperator{eltype(op)}, composite_size(op))
    for i in 1:composite_size(op, 1)
        for j in 1:composite_size(op, 2)
            operators[i,j] = ZeroOperator(src(op, j), dest(op, i))
        end
    end
    BlockOperator(operators)
end

# Return the source of the i-th column
src(op::BlockOperator, j) = j==1 && is_columnlike(op) ? src(op) : element(src(op), j)

# Return the destination of the i-th row
dest(op::BlockOperator, i) = i==1 && is_rowlike(op) ? dest(op) : element(dest(op), i)

element(op::BlockOperator, i::Int, j::Int) = op.operators[i,j]
elements(op::BlockOperator) = op.operators

composite_size(op::BlockOperator) = size(op.operators)

composite_size(op::BlockOperator, dim) = size(op.operators, dim)

is_rowlike(op::BlockOperator) = size(op.operators,1) == 1

is_columnlike(op::BlockOperator) = size(op.operators,2) == 1

function apply!(op::BlockOperator, coef_dest, coef_src)
    if is_rowlike(op)
        apply_rowoperator!(op, coef_dest, coef_src, op.scratch_src, op.scratch_dest)
    elseif is_columnlike(op)
        apply_columnoperator!(op, coef_dest, coef_src, op.scratch_src, op.scratch_dest)
    else
        apply_block_operator!(op, coef_dest, coef_src, op.scratch_src, op.scratch_dest)
    end
    coef_dest
end

function apply_block_operator!(op::BlockOperator, coef_dest::AbstractVector, coef_src::AbstractVector, scratch_src, scratch_dest)
    delinearize_coefficients!(scratch_src, coef_src)
    apply_block_operator!(op, scratch_dest, scratch_src, scratch_src, scratch_dest)
    linearize_coefficients!(coef_dest, scratch_dest)
end

function apply_block_operator!(op::BlockOperator, coef_dest::AbstractVector, coef_src::MultiArray, scratch_src, scratch_dest)
    apply_block_operator!(op, scratch_dest, coef_src, scratch_src, scratch_dest)
    linearize_coefficients!(coef_dest, scratch_dest)
end

function apply_block_operator!(op::BlockOperator, coef_dest::MultiArray, coef_src::AbstractVector, scratch_src, scratch_dest)
    delinearize_coefficients!(scratch_src, coef_src)
    apply_block_operator!(op, coef_dest, scratch_src, scratch_src, scratch_dest)
end

function apply_block_operator!(op::BlockOperator, coef_dest::MultiArray, coef_src::MultiArray, scratch_src, scratch_dest)
    for m in 1:nb_elements(coef_dest)
        fill!(element(coef_dest, m), 0)
        for n in 1:nb_elements(coef_src)
            apply_block_element!(element(op, m, n), element(coef_dest, m),
                element(coef_src,n), element(op.scratch_dest, m))
        end
    end
    coef_dest
end

function apply_block_element!(op, coef_dest, coef_src, scratch)
    apply!(op, scratch, coef_src)
    for i in eachindex(coef_dest)
        coef_dest[i] += scratch[i]
    end
end

function apply_rowoperator!(op::BlockOperator, coef_dest, coef_src::MultiArray, scratch_src, scratch_dest)
    fill!(coef_dest, 0)
    for n in 1:nb_elements(coef_src)
        apply!(op.operators[1,n], scratch_dest, element(coef_src,n))
        for i in eachindex(coef_dest)
            coef_dest[i] += scratch_dest[i]
        end
    end
end

function apply_rowoperator!(op::BlockOperator, coef_dest, coef_src::AbstractVector, scratch_src, scratch_dest)
    delinearize_coefficients!(scratch_src, coef_src)
    apply_rowoperator!(op, coef_dest, scratch_src, scratch_src, scratch_dest)
end

function apply_columnoperator!(op::BlockOperator, coef_dest::MultiArray, coef_src, scratch_src, scratch_dest)
    for m in 1:nb_elements(coef_dest)
        apply!(op.operators[m,1], element(coef_dest, m), coef_src)
    end
end

function apply_columnoperator!(op::BlockOperator, coef_dest::AbstractVector, coef_src, scratch_src, scratch_dest)
    apply_columnoperator!(op, scratch_dest, coef_src, scratch_src, scratch_dest)
    linearize_coefficients!(coef_dest, scratch_dest)
end

hcat(op1::AbstractOperator, op2::AbstractOperator) = block_row_operator(op1, op2)
vcat(op1::AbstractOperator, op2::AbstractOperator) = block_column_operator(op1, op2)

hcat(op1::BlockOperator, op2::BlockOperator) = BlockOperator(hcat(op1.operators, op2.operators))
vcat(op1::BlockOperator, op2::BlockOperator) = BlockOperator(vcat(op1.operators, op2.operators))

ctranspose(op::BlockOperator) = BlockOperator(ctranspose(op.operators))



############################
# Block diagonal operators
############################

"""
A BlockDiagonalOperator has a block matrix structure like a BlockOperator, but
with only blocks on the diagonal.
"""
struct BlockDiagonalOperator{T} <: ParentOperator{T}
    operators   ::  Array{AbstractOperator{T}, 1}
    src         ::  Span
    dest        ::  Span
end

function BlockDiagonalOperator{O<:AbstractOperator}(operators::Array{O,1}, src, dest)
  T = promote_type(map(eltype, operators)...)
  BlockDiagonalOperator{T}(operators, src, dest)
end

BlockDiagonalOperator{O<:AbstractOperator}(operators::Array{O,1}) =
    BlockDiagonalOperator(operators, multispan(map(src, operators)), multispan(map(dest, operators)))

operators(op::BlockDiagonalOperator) = op.operators
elements(op::BlockDiagonalOperator) = op.operators

function block_operator(op::BlockDiagonalOperator)
    ops = Array(AbstractOperator{eltype(op)}, nb_elements(op), nb_elements(op))
    n = length(op.operators)
    for i in 1:n,j in 1:n
        ops[i,j] = element(op, i, j)
    end
    BlockOperator(ops)
end

composite_size(op::BlockDiagonalOperator) = (l = length(op.operators); (l,l))

block_diagonal_operator(op1::AbstractOperator, op2::AbstractOperator) =
    BlockDiagonalOperator([op1,op2])
block_diagonal_operator(op1::BlockDiagonalOperator, op2::BlockDiagonalOperator) =
    BlockDiagonalOperator(vcat(elements(op1),elements(op2)))
block_diagonal_operator(op1::BlockDiagonalOperator, op2::AbstractOperator) =
    BlockDiagonalOperator(vcat(elements(op1), op2))
block_diagonal_operator(op1::AbstractOperator, op2::BlockDiagonalOperator) =
    BlockDiagonalOperator(vcat(op1,elements(op2)))

function block_diagonal_operator(op1::AbstractOperator, op2::BlockOperator)
    ops = Array(AbstractOperator{eltype(op1)}, 1+composite_size(op2,1), 1+composite_size(op2,2))
    ops[1,1] = op1
    for i in 1:composite_size(op2,1)
        ops[1+i,1] = ZeroOperator(src(op1), dest(op2,i))
    end
    for j in 1:composite_size(op2,2)
        ops[1,1+j] = ZeroOperator(src(op2,j), dest(op1))
    end
    ops[2:end,2:end] = op2.operators
    BlockOperator(ops)
end

function block_diagonal_operator(ops::AbstractArray{AbstractOperator{T}}) where {T}
    @assert length(ops) > 1
    if length(ops) > 2
        block_diagonal_operator(ops[1], ops[2], ops[2:end])
    else
        block_diagonal_operator(ops[1], ops[2])
    end
end

⊕(op1::AbstractOperator, op2::AbstractOperator) = block_diagonal_operator(op1, op2)

function element(op::BlockDiagonalOperator{T}, i::Int, j::Int) where {T}
    if i == j
        op.operators[i]
    else
        ZeroOperator{T}(element(op.src, j), element(op.dest, i))
    end
end

apply!(op::BlockDiagonalOperator, coef_dest::MultiArray, coef_src::Array{T,1}) where {T} =
    apply!(op, coef_dest, delinearize_coefficients(coef_dest, coef_src))

function apply!(op::BlockDiagonalOperator, coef_dest::MultiArray, coef_src::MultiArray)
    for i in 1:nb_elements(coef_src)
        apply!(op.operators[i], element(coef_dest, i), element(coef_src, i))
    end
    coef_dest
end

ctranspose(op::BlockDiagonalOperator) = BlockDiagonalOperator(map(ctranspose, operators(op)))

inv(op::BlockDiagonalOperator) = BlockDiagonalOperator(map(inv, operators(op)), dest(op), src(op))
# inv(op::BlockDiagonalOperator) = BlockDiagonalOperator(AbstractOperator{eltype(op)}[inv(o) for o in BasisFunctions.operators(op)])

function stencil(op::BlockOperator)
    A = Any[]
    push!(A,"[")
    for i=1:size(elements(op),1)
        i!=1 && push!(A,";\t")
        push!(A,element(op,i,1))
        for j=2:size(elements(op),2)
            push!(A,", \t")
            push!(A,element(op,i,j))
        end
    end
    push!(A,"]")
    A
end

function stencil(op::BlockDiagonalOperator)
    A = Any[]
    push!(A,"[")
    for i=1:composite_size(op)[1]
        i!=1 && push!(A,";\t 0")
        i==1 && push!(A,element(op,1,1))
        for j=2:composite_size(op)[2]
            push!(A,", \t")
            i==j && push!(A,element(op,i,j))
            i!=j && push!(A,"0")
        end
    end
    push!(A,"]")
    A
end
