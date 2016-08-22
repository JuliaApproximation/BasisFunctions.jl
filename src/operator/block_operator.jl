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
    ops     ::  Array{AbstractOperator{ELT}, 2}
    src     ::  FunctionSet
    dest    ::  FunctionSet

    # scratch_dest holds memory for each subset of the destination set (which is
    # a multiset).
    scratch_dest    ::  Array{Array{ELT},1}

    function BlockOperator(ops, src, dest)
        scratch_dest = Array(Array{ELT}, size(ops,1))
        for i in 1:length(scratch_dest)
            scratch_dest[i] = zeros(ELT, size(element(dest,i)))
        end
        new(ops, src, dest, scratch_dest)
    end
end

function BlockOperator{ELT}(ops::Array{AbstractOperator{ELT},2})
    # Avoid 1x1 block operators
    @assert size(ops,1) + size(ops,2) > 2

    if size(ops,2) > 1
        op_src = MultipleSet(map(src, ops[1,:]))
    else
        op_src = src(ops[1,1])
    end
    if size(ops,1) > 1
        op_dest = MultipleSet(map(dest, ops[:,1]))
    else
        op_dest = dest(ops[1,1])
    end
    BlockOperator{ELT}(ops, op_src, op_dest)
end

element(op::BlockOperator, i::Int, j::Int) = op.ops[i,j]

is_rowlike(op::BlockOperator) = size(op.ops,1) == 1

is_columnlike(op::BlockOperator) = size(op.ops,2) == 1

function apply!(op::BlockOperator, coef_dest, coef_src)
    if is_rowlike(op)
        apply_rowoperator!(op, coef_dest, coef_src)
    elseif is_columnlike(op)
        apply_columnoperator!(op, coef_dest, coef_src)
    else
        for m in 1:length(coef_dest)
            coef_dest[m][:] = 0
            for n in 1:length(coef_src)
                apply_block_element!(element(op, m, n), coef_dest[m], coef_src[n], op.scratch_dest[m])
            end
        end
    end
    coef_dest
end

function apply_block_element!(op, coef_dest, coef_src, scratch_dest)
    apply!(op, scratch_dest, coef_src)
    coef_dest += scratch_dest
end

function apply_rowoperator!(op::BlockOperator, coef_dest, coef_src)
    coef_dest[:] = 0
    for n in 1:length(coef_src)
        apply!(op.ops[1,n], scratch_dest[1], coef_src[n])
        coef_dest += scratch_dest[1]
    end
end

function apply_columnoperator!(op::BlockOperator, coef_dest, coef_src)
    for m in 1:length(coef_dest)
        apply!(op.ops[m,1], coef_dest[m], coef_src)
    end
end


"""
A BlockDiagonalOperator has a block matrix structure like a BlockOperator, but
with only blocks on the diagonal.
"""
immutable BlockDiagonalOperator{ELT} <: AbstractOperator{ELT}
    ops     ::  Array{AbstractOperator{ELT}, 1}
    src     ::  FunctionSet
    dest    ::  FunctionSet
end

function BlockDiagonalOperator{ELT}(ops::Array{AbstractOperator{ELT},1})
    op_src = MultipleSet(map(src, ops))
    op_dest = MultipleSet(map(dest, ops))
    BlockDiagonalOperator{ELT}(ops, op_src, op_dest)
end

function element{ELT}(op::BlockDiagonalOperator{ELT}, i::Int, j::Int)
    if i == j
        op.ops[i]
    else
        ZeroOperator{ELT}(element(op.src, j), element(op.src, i))
    end
end

function apply!(op::BlockDiagonalOperator, coef_dest, coef_src)
    for i in 1:length(coef_src)
        apply!(op.ops[i], coef_dest[i], coef_src[i])
    end
    coef_dest
end
