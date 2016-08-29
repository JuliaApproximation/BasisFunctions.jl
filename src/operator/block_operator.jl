# block_operator.jl

"""
A BlockOperator has a block matrix structure, where each block is an operator,
and it acts on multiple sets.

A BlockOperator is row-like if it only has one row of blocks. In that case, the
destination of the operator is not necessarily a multiset.

A BlockOperator is column-like if it only has one column of blocks. In that case,
the source set of the operator is not necessarily a multiset.
"""
immutable BlockOperator{ELT} <: AbstractOperator{ELT}
    operators   ::  Array{AbstractOperator{ELT}, 2}
    src         ::  FunctionSet
    dest        ::  FunctionSet

    # scratch_dest holds memory for each subset of the destination set (which is
    # a multiset).
    scratch_dest

    function BlockOperator(operators, src, dest)
        scratch_dest = zeros(ELT, dest)
        new(operators, src, dest, scratch_dest)
    end
end

function BlockOperator{OP <: AbstractOperator}(operators::Array{OP,2})
    # Avoid 1x1 block operators
    @assert size(operators,1) + size(operators,2) > 2

    op_src = multiset(map(src, operators[1,:]))
    op_dest = multiset(map(dest, operators[:,1]))
    ELT = promote_type(eltype(op_src), eltype(op_dest))
    BlockOperator{ELT}(operators, op_src, op_dest)
end

function block_row_operator{OP <: AbstractOperator}(ops::Array{OP, 1})
    ELT = eltype(ops[1])
    operators = Array(AbstractOperator{ELT}, 1, length(ops))
    operators[:] = ops
    BlockOperator(operators)
end

function block_column_operator{OP <: AbstractOperator}(ops::Array{OP, 1})
    ELT = eltype(ops[1])
    operators = Array(AbstractOperator{ELT}, length(ops), 1)
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

composite_size(op::BlockOperator) = size(op.operators)

composite_size(op::BlockOperator, dim) = size(op.operators, dim)

is_rowlike(op::BlockOperator) = size(op.operators,1) == 1

is_columnlike(op::BlockOperator) = size(op.operators,2) == 1

function apply!(op::BlockOperator, coef_dest, coef_src)
    if is_rowlike(op)
        apply_rowoperator!(op, coef_dest, coef_src, op.scratch_dest)
    elseif is_columnlike(op)
        apply_columnoperator!(op, coef_dest, coef_src)
    else
        for m in 1:composite_length(coef_dest)
            fill!(element(coef_dest, m), 0)
            for n in 1:composite_length(coef_src)
                apply_block_element!(element(op, m, n), element(coef_dest, m),
                    element(coef_src,n), element(op.scratch_dest, m))
            end
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

function apply_rowoperator!(op::BlockOperator, coef_dest, coef_src::MultiArray, scratch)
    fill!(coef_dest, 0)
    for n in 1:composite_length(coef_src)
        apply!(op.operators[1,n], scratch, element(coef_src,n))
        for i in eachindex(coef_dest)
            coef_dest[i] += scratch[i]
        end
    end
end

function apply_columnoperator!(op::BlockOperator, coef_dest::MultiArray, coef_src)
    for m in 1:composite_length(coef_dest)
        apply!(op.operators[m,1], element(coef_dest, m), coef_src)
    end
end


hcat(op1::AbstractOperator, op2::AbstractOperator) = block_row_operator([op1,op2])
vcat(op1::AbstractOperator, op2::AbstractOperator) = block_column_operator([op1,op2])

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
immutable BlockDiagonalOperator{ELT} <: AbstractOperator{ELT}
    operators   ::  Array{AbstractOperator{ELT}, 1}
    src         ::  FunctionSet
    dest        ::  FunctionSet
end

function BlockDiagonalOperator{ELT}(operators::Array{AbstractOperator{ELT},1})
    op_src = MultiSet(map(src, operators))
    op_dest = MultiSet(map(dest, operators))
    BlockDiagonalOperator{ELT}(operators, op_src, op_dest)
end

operators(op::BlockDiagonalOperator) = op.operators

function block_operator(op::BlockDiagonalOperator)
    ops = Array(AbstractOperator{eltype(op)}, composite_length(op), composite_length(op))
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

function block_diagonal_operator{ELT}(ops::AbstractArray{AbstractOperator{ELT}})
    @assert length(ops) > 1
    if length(ops) > 2
        block_diagonal_operator(ops[1], ops[2], ops[2:end])
    else
        block_diagonal_operator(ops[1], ops[2])
    end
end

⊕(op1::AbstractOperator, op2::AbstractOperator) = block_diagonal_operator(op1, op2)

function element{ELT}(op::BlockDiagonalOperator{ELT}, i::Int, j::Int)
    if i == j
        op.operators[i]
    else
        ZeroOperator{ELT}(element(op.src, j), element(op.dest, i))
    end
end

function apply!(op::BlockDiagonalOperator, coef_dest::MultiArray, coef_src::MultiArray)
    for i in 1:composite_length(coef_src)
        apply!(op.operators[i], element(coef_dest, i), element(coef_src, i))
    end
    coef_dest
end

ctranspose(op::BlockDiagonalOperator) = BlockDiagonalOperator(map(ctranspose, operators(op)))

inv(op::BlockDiagonalOperator) = BlockDiagonalOperator(map(inv, operators(op)))
