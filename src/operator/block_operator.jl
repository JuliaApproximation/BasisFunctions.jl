
"""
A BlockOperator has a block matrix structure, where each block is an operator,
and it acts on multiple sets.

A BlockOperator is row-like if it only has one row of blocks. In that case, the
destination of the operator is not necessarily a multidict.

A BlockOperator is column-like if it only has one column of blocks. In that case,
the source set of the operator is not necessarily a multidict.
"""
struct BlockOperator{T} <: DictionaryOperator{T}
    operators   ::  AbstractArray{DictionaryOperator{T}, 2}
    src         ::  Dictionary
    dest        ::  Dictionary

    # scratch_src and scratch_dest hold scratch memory for each subset of the
    # source and destination sets, for allocation-free implementation of the
    # action of the operator
    scratch_src
    scratch_dest

    function BlockOperator{T}(operators, src, dest) where {T}
        # Avoid 1x1 block operators
        @assert size(operators,1) + size(operators,2) > 2
        scratch_src = zeros(src)
        scratch_dest = zeros(dest)
        new(operators, src, dest, scratch_src, scratch_dest)
    end
end

function BlockOperator(operators::AbstractArray{OP,2},
            op_src = multidict(map(src, operators[1,:])...),
            op_dest = multidict(map(dest, operators[:,1])...)) where {OP <: DictionaryOperator}
    T = promote_type(operatoreltype(op_src, op_dest), map(eltype, operators)...)
    BlockOperator{T}(operators, op_src, op_dest)
end

function ArrayOperator(A::BlockArray{T}, src::Dictionary, dest::Dictionary) where {T}
    elements_src = iscomposite(src) ? elements(src) : (src,)
    elements_dest = iscomposite(dest) ? elements(dest) : (dest,)
    BlockOperator{T}([ArrayOperator(view(A, Block(i, j)), srcj, desti) for (i,desti) in enumerate(elements_dest), (j,srcj) in enumerate(elements_src)], src, dest)
end


# For legacy reasons
function block_row_operator(op1::DictionaryOperator, op2::DictionaryOperator, dicts::Dictionary...)
    T = promote_type(eltype(op1), eltype(op2), operatoreltype(dicts...))
    operators = Array{DictionaryOperator{T}}(1, 2)
    operators[1] = op1
    operators[2] = op2
    BlockOperator(operators, dicts...)
end

block_row_operator(ops::DictionaryOperator...) = block_row_operator(collect(ops))
function block_row_operator(ops::AbstractArray{<:DictionaryOperator,1}, dicts...)
    T = promote_type(map(eltype, ops)...)
    operators = Array{DictionaryOperator{T}}(undef, 1, length(ops))
    operators[:] = ops
    BlockOperator(operators, dicts...)
end


block_column_operator(ops::DictionaryOperator...) = block_column_operator(collect(ops))
function block_column_operator(ops::AbstractArray{<:DictionaryOperator,1}, dicts...)
    T = promote_type(map(eltype, ops)...)
    operators = Array{DictionaryOperator{T}}(undef, length(ops), 1)
    operators[:] = ops
    BlockOperator(operators, dicts...)
end

unsafe_wrap_operator(src, dest, op::BlockOperator) = similar_operator(op, src, dest)

similar_operator(op::BlockOperator{T}, src, dest) where {T} =
    BlockOperator{T}(op.operators, src, dest)

# Return a block operator the size of the given operator, but filled with zero
# operators.
function zeros(op::BlockOperator)
    operators = zeros(DictionaryOperator{eltype(op)}, composite_size(op))
    for i in 1:composite_size(op, 1)
        for j in 1:composite_size(op, 2)
            operators[i,j] = ZeroOperator(src(op, j), dest(op, i))
        end
    end
    BlockOperator(operators)
end

# Return the source of the i-th column
src(op::BlockOperator, j) = j==1 && iscolumnlike(op) ? src(op) : element(src(op), j)

# Return the destination of the i-th row
dest(op::BlockOperator, i) = i==1 && isrowlike(op) ? dest(op) : element(dest(op), i)

element(op::BlockOperator, i::Int, j::Int) = op.operators[i,j]
elements(op::BlockOperator) = op.operators

composite_size(op::BlockOperator) = size(op.operators)

composite_size(op::BlockOperator, dim) = size(op.operators, dim)

isrowlike(op::BlockOperator) = size(op.operators,1) == 1

iscolumnlike(op::BlockOperator) = size(op.operators,2) == 1

function apply!(op::BlockOperator, coef_dest, coef_src)
    if isrowlike(op)
        apply_rowoperator!(op, coef_dest, coef_src, op.scratch_src, op.scratch_dest)
    elseif iscolumnlike(op)
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

function apply_block_operator!(op::BlockOperator, coef_dest::AbstractVector, coef_src::BlockVector, scratch_src, scratch_dest)
    apply_block_operator!(op, scratch_dest, coef_src, scratch_src, scratch_dest)
    linearize_coefficients!(coef_dest, scratch_dest)
end

function apply_block_operator!(op::BlockOperator, coef_dest::BlockVector, coef_src::AbstractVector, scratch_src, scratch_dest)
    delinearize_coefficients!(scratch_src, coef_src)
    apply_block_operator!(op, coef_dest, scratch_src, scratch_src, scratch_dest)
end

function apply_block_operator!(op::BlockOperator, coef_dest::BlockVector, coef_src::BlockVector, scratch_src, scratch_dest)
    for m in 1:numelements(coef_dest)
        fill!(element(coef_dest, m), 0)
        for n in 1:numelements(coef_src)
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

function apply_rowoperator!(op::BlockOperator, coef_dest, coef_src::BlockVector, scratch_src, scratch_dest)
    fill!(coef_dest, 0)
    for n in 1:numelements(coef_src)
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

function apply_columnoperator!(op::BlockOperator, coef_dest::BlockVector, coef_src, scratch_src, scratch_dest)
    for m in 1:numelements(coef_dest)
        apply!(op.operators[m,1], element(coef_dest, m), coef_src)
    end
end

function apply_columnoperator!(op::BlockOperator, coef_dest::AbstractVector, coef_src, scratch_src, scratch_dest)
    apply_columnoperator!(op, scratch_dest, coef_src, scratch_src, scratch_dest)
    linearize_coefficients!(coef_dest, scratch_dest)
end

hcat(ops::DictionaryOperator...) = block_row_operator(ops...)
vcat(ops::DictionaryOperator...) = block_column_operator(ops...)

function hvcat(rows::Tuple{Vararg{Int}}, ops::DictionaryOperator...)
    T = promote_type(map(eltype, ops)...)
    operators = Array{DictionaryOperator{T}}(undef, floor(Int,length(ops)/rows[1]), rows[1])
    for i in 1:length(operators)
        operators[i] = ops[i]
    end
    BlockOperator(operators)
end

hcat(op1::BlockOperator, op2::BlockOperator) = BlockOperator(hcat(op1.operators, op2.operators))
vcat(op1::BlockOperator, op2::BlockOperator) = BlockOperator(vcat(op1.operators, op2.operators))

adjoint(op::BlockOperator) = BlockOperator(Array{DictionaryOperator}(adjoint(op.operators)), dest(op), src(op))
conj(op::BlockOperator) = BlockOperator(Array{DictionaryOperator}(conj(op.operators)), src(op), dest(op))



############################
# Block diagonal operators
############################

"""
A BlockDiagonalOperator has a block matrix structure like a BlockOperator, but
with only blocks on the diagonal.
"""
struct BlockDiagonalOperator{T} <: DictionaryOperator{T}
    operators   ::  AbstractArray{DictionaryOperator{T}, 1}
    src         ::  Dictionary
    dest        ::  Dictionary
end

function BlockDiagonalOperator(operators::AbstractArray{O,1}, src, dest) where {O<:DictionaryOperator}
    T = promote_type(operatoreltype(src,dest), map(eltype, operators)...)
    BlockDiagonalOperator{T}(operators, src, dest)
end

function BlockDiagonalOperator(operators::AbstractArray{O,1}) where {O<:DictionaryOperator}
    BlockDiagonalOperator(operators, multidict(map(src, operators)), multidict(map(dest, operators)))
end

operators(op::BlockDiagonalOperator) = op.operators
elements(op::BlockDiagonalOperator) = op.operators

function block_operator(op::BlockDiagonalOperator)
    ops = Array(DictionaryOperator{eltype(op)}, numelements(op), numelements(op))
    n = length(op.operators)
    for i in 1:n,j in 1:n
        ops[i,j] = element(op, i, j)
    end
    BlockOperator(ops)
end

composite_size(op::BlockDiagonalOperator) = (l = length(op.operators); (l,l))

block_diagonal_operator(op1::DictionaryOperator, op2::DictionaryOperator) =
    BlockDiagonalOperator([op1,op2])
block_diagonal_operator(op1::BlockDiagonalOperator, op2::BlockDiagonalOperator) =
    BlockDiagonalOperator(vcat(elements(op1),elements(op2)))
block_diagonal_operator(op1::BlockDiagonalOperator, op2::DictionaryOperator) =
    BlockDiagonalOperator(vcat(elements(op1), op2))
block_diagonal_operator(op1::DictionaryOperator, op2::BlockDiagonalOperator) =
    BlockDiagonalOperator(vcat(op1,elements(op2)))

function block_diagonal_operator(op1::DictionaryOperator, op2::BlockOperator)
    ops = Array(DictionaryOperator{eltype(op1)}, 1+composite_size(op2,1), 1+composite_size(op2,2))
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

function block_diagonal_operator(ops::AbstractArray{DictionaryOperator{T}}) where {T}
    @assert length(ops) > 1
    if length(ops) > 2
        block_diagonal_operator(ops[1], ops[2], ops[2:end])
    else
        block_diagonal_operator(ops[1], ops[2])
    end
end

⊕(op1::DictionaryOperator, op2::DictionaryOperator) = block_diagonal_operator(op1, op2)

function element(op::BlockDiagonalOperator{T}, i::Int, j::Int) where {T}
    if i == j
        op.operators[i]
    else
        ZeroOperator{T}(element(op.src, j), element(op.dest, i))
    end
end

apply!(op::BlockDiagonalOperator, coef_dest::BlockVector, coef_src::Array{T,1}) where {T} =
    apply!(op, coef_dest, delinearize_coefficients(coef_dest, coef_src))

function apply!(op::BlockDiagonalOperator, coef_dest::BlockVector, coef_src::BlockVector)
    for i in 1:numelements(coef_src)
        apply!(op.operators[i], element(coef_dest, i), element(coef_src, i))
    end
    coef_dest
end

adjoint(op::BlockDiagonalOperator) = BlockDiagonalOperator(map(adjoint, operators(op)), dest(op), src(op))
conj(op::BlockDiagonalOperator) = BlockDiagonalOperator(map(conj, operators(op)), src(op), dest(op))

inv(op::BlockDiagonalOperator) = BlockDiagonalOperator(map(inv, operators(op)), dest(op), src(op))
# inv(op::BlockDiagonalOperator) = BlockDiagonalOperator(DictionaryOperator{eltype(op)}[inv(o) for o in BasisFunctions.operators(op)])

function stencilarray(op::BlockOperator)
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

function stencilarray(op::BlockDiagonalOperator)
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

Matrix(op::Union{BlockOperator,BlockDiagonalOperator}) =
    Matrix(matrix(op))

function matrix(op::Union{BlockOperator,BlockDiagonalOperator})
    cs = composite_size(op)
    dest_cs = cs[1] == 1 ? [length(dest(op))] :  [map(length, elements(dest(op)))...]
    src_cs = cs[2] == 1 ? [length(src(op))] :  [map(length, elements(src(op)))...]
    a = BlockArray{eltype(op)}(undef_blocks, dest_cs, src_cs)
    matrix!(op, a)
    a
end

function matrix!(op::Union{BlockOperator,BlockDiagonalOperator}, a)
    for i in 1:composite_size(op)[1]
        for j in 1:composite_size(op)[2]
            a[Block(i,j)] = Matrix(element(op, i, j))
        end
    end
end
