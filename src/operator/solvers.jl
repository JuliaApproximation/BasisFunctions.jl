
"""
Supertype of all solver operators. A solver operator typically implements an
efficient algorithm to apply the inverse of an operator. Examples include
a solver based on QR or SVD factorizations.

A solver operator stores the operator it has inverted.
"""
abstract type AbstractSolverOperator{T} <: DictionaryOperator{T}
end

operator(op::AbstractSolverOperator) = op.op

src(op::AbstractSolverOperator) = dest(operator(op))

dest(op::AbstractSolverOperator) = src(operator(op))

inv(op::AbstractSolverOperator) = op.op


"A GenericSolverOperator wraps around a generic solver type."
struct GenericSolverOperator{Q,T} <: AbstractSolverOperator{T}
    op      ::  DictionaryOperator
    solver  ::  Q
    # In case the operator does not map between vectors, we allocate space
    # for two vectors so that we can convert between representations.
    src_linear  ::  Vector{T}
    dest_linear ::  Vector{T}

    GenericSolverOperator{Q,T}(op, solver) where {Q,T} =
        new{Q,T}(op, solver, zeros(T, length(dest(op))), zeros(T, length(src(op))))
end


# The solver should be the inverse of the given operator
GenericSolverOperator(op::DictionaryOperator{T}, solver) where T =
    GenericSolverOperator{typeof(solver),T}(op, solver)

similar_operator(op::GenericSolverOperator, src, dest) =
    GenericSolverOperator(similar_operator(operator(op), dest, src), op.solver)

apply!(op::GenericSolverOperator, coef_dest::Vector, coef_src::Vector) =
    _apply!(op, coef_dest, coef_src, op.solver)

function apply!(op::GenericSolverOperator, coef_dest, coef_src)
    copyto!(op.src_linear, coef_src)
    _apply!(op, op.dest_linear, op.src_linear, op.solver)
    copyto!(coef_dest, op.dest_linear)
end

function apply!(op::GenericSolverOperator, coef_dest, coef_src::Vector)
    _apply!(op, op.dest_linear, coef_src, op.solver)
    copyto!(coef_dest, op.dest_linear)
end

function apply!(op::GenericSolverOperator, coef_dest::Vector, coef_src)
    copyto!(op.src_linear, coef_src)
    _apply!(op, coef_dest, op.src_linear, op.solver)
end


# This is the generic case
function _apply!(op::GenericSolverOperator, coef_dest, coef_src, solver)
    coef_dest[:] = solver \ coef_src
    coef_dest
end

# # More efficient version for vectors and factorization solvers
# # TODO: these don't work for rectangular matrices
#     function _apply!(op::GenericSolverOperator, coef_dest::Vector, coef_src::Vector, solver::Factorization)
#         copyto!(coef_dest, coef_src)
#         ldiv!(solver, coef_dest)
#     end
#     function _apply!(op::GenericSolverOperator, coef_dest::Vector, coef_src::Vector, solver::LinearAlgebra.SVD)
#         copyto!(coef_dest, coef_src)
#         coef_dest[:] = ldiv!(solver, coef_dest)
#     end

adjoint(op::GenericSolverOperator) = @warn("not implemented")

qr_factorization(matrix) = qr(matrix, Val(true))
svd_factorization(matrix) = svd(matrix, full=false)

QR_solver(op::DictionaryOperator; options...) = GenericSolverOperator(op, qr_factorization(matrix(op)))
SVD_solver(op::DictionaryOperator; options...) = GenericSolverOperator(op, svd_factorization(matrix(op)))
