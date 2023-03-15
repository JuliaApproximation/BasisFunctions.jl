"""
Supertype of all dictionary solver operators. A solver operator typically implements an
efficient algorithm to apply the inverse of an operator. Examples include
a solver based on QR or SVD factorizations.

A solver operator stores the operator it has inverted.
"""
abstract type AbstractSolverOperator <: AbstractOperator
end

operator(op::AbstractSolverOperator) = op.op

src_space(op::AbstractSolverOperator) = dest_space(operator(op))

dest_space(op::AbstractSolverOperator) = src_space(operator(op))
dest(op::AbstractSolverOperator) = src(operator(op))

inv(op::AbstractSolverOperator) = operator(op)

"""
Supertype of all dictionary solver operators. A solver operator typically implements an
efficient algorithm to apply the inverse of an operator. Examples include
a solver based on QR or SVD factorizations.

A solver operator stores the operator it has inverted.
"""
abstract type DictionarySolverOperator{T} <: DictionaryOperator{T}
end

operator(op::DictionarySolverOperator) = op.op

src(op::DictionarySolverOperator) = dest(operator(op))

dest(op::DictionarySolverOperator) = src(operator(op))

inv(op::DictionarySolverOperator) = operator(op)


abstract type VectorizingSolverOperator{T} <: DictionarySolverOperator{T} end

apply!(op::VectorizingSolverOperator, coef_dest::Vector, coef_src::Vector) =
    linearized_apply!(op, coef_dest, coef_src)

function apply!(op::VectorizingSolverOperator, coef_dest, coef_src)
    copyto!(op.src_linear, coef_src)
    linearized_apply!(op, op.dest_linear, op.src_linear)
    copyto!(coef_dest, op.dest_linear)
end

function apply!(op::VectorizingSolverOperator, coef_dest, coef_src::Vector)
    linearized_apply!(op, op.dest_linear, coef_src)
    copyto!(coef_dest, op.dest_linear)
end

function apply!(op::VectorizingSolverOperator, coef_dest::Vector, coef_src)
    copyto!(op.src_linear, coef_src)
    linearized_apply!(op, coef_dest, op.src_linear)
end

"A GenericSolverOperator wraps around a generic solver type."
struct GenericSolverOperator{Q,T} <: VectorizingSolverOperator{T}
    op      ::  DictionaryOperator
    solver  ::  Q
    # In case the operator does not map between vectors, we allocate space
    # for two vectors so that we can convert between representations.
    src_linear  ::  Vector{T}
    dest_linear ::  Vector{T}

    GenericSolverOperator{Q,T}(op, solver) where {Q,T} =
        new{Q,T}(op, solver, zeros(T, length(dest(op))), zeros(T, length(src(op))))
end

linearized_apply!(op::GenericSolverOperator, dest_vc::Vector, src_vc::Vector) =
    linearized_apply!(op, dest_vc, src_vc, op.solver)

# The solver should be the inverse of the given operator
GenericSolverOperator(op::DictionaryOperator{T}, solver) where T =
    GenericSolverOperator{typeof(solver),T}(op, solver)

similar_operator(op::GenericSolverOperator, src, dest) =
    GenericSolverOperator(similar_operator(operator(op), dest, src), op.solver)


# This is the generic case
function linearized_apply!(op::GenericSolverOperator, coef_dest::Vector, coef_src::Vector, solver)
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

adjoint(op::GenericSolverOperator) = error("adjoint not implemented for $(op)")

if VERSION ≥ v"1.7-"
    qr_factorization(op::DictionaryOperator) = qr(matrix(op), ColumnNorm())
else
    qr_factorization(op::DictionaryOperator) = qr(matrix(op), Val(true))
end

function svd_factorization(op::DictionaryOperator)
    local F
    A = Matrix(op)
    try
        F = svd(A, full=false)
    catch
        # Sometimes svd does not work because A is a structured matrix.
        # In that case, we convert to a dense matrix.
        F = svd(collect(A), full=false)
    end
    return F
end

function regularized_svd_factorization(op::DictionaryOperator;
            verbose=false, threshold = 1000*eps(real(eltype(op))), options...)
    F = svd_factorization(op)
    U = F.U
    S = F.S
    Vt = F.Vt
    I = findfirst(F.S .< threshold)
    if I == nothing
        verbose && println("SVD: no regularization needed (threshold: $threshold)")
        return F
    else
        verbose && println("SVD truncated at rank $I of $(minimum(size(op))) (threshold: $threshold)")
        U_reg = U[:,1:I]
        Vt_reg = Vt[1:I,:]
        S_reg = S[1:I]
        return SVD(U_reg, S_reg, Vt_reg)
    end
end

QR_solver(op::DictionaryOperator; options...) =
    GenericSolverOperator(op, qr_factorization(op))
SVD_solver(op::DictionaryOperator; options...) =
    GenericSolverOperator(op, svd_factorization(op))

regularized_SVD_solver(op::DictionaryOperator; options...) =
    GenericSolverOperator(op, regularized_svd_factorization(op; options...))

for (damp, LS) in ((:λ, :LSMR), (:damp, :LSQR))
    OPTIONS = Symbol(LS,"Options")
    @eval begin
        export $OPTIONS
        mutable struct $OPTIONS{T}
            atol::T
            btol::T
            conlim::T
            $damp::T
            maxiter::Int
            verbose::Bool

            function $OPTIONS{T}(;atol=T(1e-6), btol=T(1e-6), conlim=T(Inf), λ=T(0), maxiter=-1, verbose=false) where T
                new{T}(atol, btol, conlim, λ, maxiter, verbose)
            end
            function $OPTIONS(;atol=1e-6, btol=1e-6, conlim=Inf, λ=0, maxiter=-1, verbose=false)
                T = promote_type(map(typeof, (atol, btol, conlim, λ))...)
                new{T}(T(atol), T(btol), T(conlim), T(λ), maxiter, verbose)
            end
            $OPTIONS{T}(itoptions::$OPTIONS{T}) where T =
                itoptions
            $OPTIONS{T}(itoptions::$OPTIONS) where T =
                $OPTIONS{T}(;convert(NamedTuple, itoptions)...)
        end

        convert(::Type{<:NamedTuple}, opts::$OPTIONS) =
            (atol=real(opts.atol), btol=real(opts.btol), conlim=real(opts.conlim), $damp=real(opts.$damp), verbose=real(opts.verbose), maxiter=real(opts.maxiter))
    end
end

export ItOptions
const ItOptions = LSMROptions
LSQROptions{T}(options::LSMROptions) where T =
    LSQROptions{T}(;convert(NamedTuple,options)...)

struct LSQR_solver{T} <: VectorizingSolverOperator{T}
    op      ::  DictionaryOperator{T}
    options ::  LSQROptions{T}

    src_linear  ::  Vector{T}
    dest_linear ::  Vector{T}
    LSQR_solver(op::DictionaryOperator{T}; itoptions=LSQROptions{T}(), options...) where T =
        LSQR_solver(op, LSQROptions{T}(itoptions); options...)


    function LSQR_solver(op::DictionaryOperator{T}, opts::LSQROptions{T}; verbose=false, options...) where T
        if opts.maxiter == -1
             opts.maxiter=100*max(size(op)...)
        end
        opts.verbose = verbose
        new{T}(op, opts,
            Vector{T}(undef, length(dest(op))), Vector{T}(undef, length(src(op)))
        )
    end
end

function linearized_apply!(op::LSQR_solver, coef_dest::Vector, coef_src::Vector)
    keys = convert(NamedTuple, op.options)
    op.options.verbose && @info "LSQR with $keys"
    copy!(coef_dest, lsqr(op.op, coef_src; keys...))
end


struct LSMR_solver{T} <: VectorizingSolverOperator{T}
    op      ::  DictionaryOperator{T}
    options ::  LSMROptions{T}

    src_linear  ::  Vector{T}
    dest_linear ::  Vector{T}

    LSMR_solver(op::DictionaryOperator{T}; itoptions=LSMROptions{T}(), options...) where T =
        LSMR_solver(op, LSMROptions{T}(itoptions); options...)

    function LSMR_solver(op::DictionaryOperator{T}, opts::LSMROptions{T}; verbose=false, options...) where T
        if opts.maxiter == -1
             opts.maxiter=100*max(size(op)...)
        end
        opts.verbose = verbose
        new{T}(op, opts,
            Vector{T}(undef, length(dest(op))), Vector{T}(undef, length(src(op)))
        )
    end
end

function linearized_apply!(op::LSMR_solver, coef_dest::Vector, coef_src::Vector)
    keys = convert(NamedTuple, op.options)
    op.options.verbose && @info "LSMR with $keys"
    copy!(coef_dest, lsmr(op.op, coef_src; keys...))
end
