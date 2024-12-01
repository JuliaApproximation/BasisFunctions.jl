
isequaldict1(b1::Legendre, b2::AbstractJacobi) =
    length(b1)==length(b2) && jacobi_α(b2) == 0 && jacobi_β(b2) == 0
isequaldict2(b1::AbstractJacobi, b2::Legendre) = isequaldict1(b2, b1)

isequaldict1(b1::ChebyshevU, b2::Ultraspherical) = length(b1)==length(b2) && ultraspherical_λ(b2) == 1
isequaldict2(b1::Ultraspherical, b2::ChebyshevU) = isequaldict1(b2, b1)

isequaldict1(b1::Ultraspherical{T}, b2::Jacobi) where T =
    length(b1)==length(b2) && ultraspherical_λ(b1) == one(T)/2 && (jacobi_α(b2) == jacobi_β(b2) == 0)
isequaldict2(b1::Jacobi, b2::Ultraspherical) = isequaldict1(b2, b1)

const RECURRENCEBASIS = Union{Monomials,OrthogonalPolynomials}

function conversion1(::Type{T}, src::RECURRENCEBASIS, dest::RECURRENCEBASIS; options...) where T
    if length(dest) < length(src)
        throw(ArgumentError("Cannot convert to a smaller basis."))
    end
    if length(dest) > length(src)
        n = length(src)
        dest1 = dest[1:n]
        E = extension_operator(T, dest1, dest)
        E * conversion(T, src, dest1; options...)
    else
        if isequaldict(src, dest)
            IdentityOperator(src)
        else
            conversion_using_recurrences(T, src, dest; options...)
        end
    end
end

conversion_using_recurrences(::Type{T}, src::RECURRENCEBASIS, dest::RECURRENCEBASIS; options...) where T =
    generic_conversion_using_recurrences(T, src, dest; options...)

# For two bases satisfying a recurrence relation we use that relation to compute the connection
# coefficients to transform between them.
function generic_conversion_using_recurrences(::Type{T}, src::RECURRENCEBASIS, dest::RECURRENCEBASIS; options...) where T
    @assert length(dest) == length(src)
    n = length(dest)
    A = zeros(T, n, n)
    if n > 0
        A[1,1] = 1
    end
    A0 = rec_An(src, 0)
    B0 = rec_Bn(src, 0)
    Ahat0 = rec_An(dest, 0)
    Bhat0 = rec_Bn(dest, 0)
    if n > 1
        A[1,2] = (B0 - A0*Bhat0/Ahat0)*A[1,1]
        A[2,2] = A0 / Ahat0 * A[1,1]
    end
    for k in 1:n-2
        K = k+1     # k starts at 0, K starts at 1
        Ak = rec_An(src, k)
        Bk = rec_Bn(src, k)
        Ck = rec_Cn(src, k)
        Ahatk = rec_An(dest, k)
        Bhatk = rec_Bn(dest, k)
        Chatk = rec_Bn(dest, k)
        Ahat1 = rec_An(dest, 1)
        Chat1 = rec_Cn(dest, 1)
        # j = 0
        J = 1       # j starts at 0, J starts at 1
        A[J,K+1] = Bk*A[J,K] - Ck*A[J,K-1] - Ak*Bhat0/Ahat0*A[J,K] + Ak/Ahat1*Chat1*A[J+1,K]
        for j in 1:k-1
            J = j+1
            Ahatj = rec_An(dest, j)
            Ahatjm1 = rec_An(dest, j-1)
            Ahatjp1 = rec_An(dest, j+1)
            Bhatj = rec_Bn(dest, j)
            Chatjp1 = rec_Cn(dest, j+1)
            A[J,K+1] = Bk*A[J,K] - Ck*A[J,K-1] - Ak*Bhatj/Ahatj*A[J,K] + Ak/Ahatjm1 * A[J-1,K] + Ak/Ahatjp1 * Chatjp1* A[J+1,K]
        end
        # j = k
        J = k+1
        Ahatkm1 = rec_An(dest, k-1)
        A[J,K+1] = (Bk-Ak*Bhatk/Ahatk)*A[K,K] + Ak/Ahatkm1*A[K-1,K]
        # j = k+1
        J = k+2
        A[J,K+1] = Ak / Ahatk*A[K,K]
    end
    ArrayOperator(UpperTriangular(A), src, dest)
end


function conversion_using_recurrences(::Type{T}, src::Jacobi, dest::Ultraspherical; options...) where T
    @assert length(src) == length(dest)
    if (jacobi_α(src) == jacobi_β(src)) && (jacobi_α(src) ≈ jacobi_α(dest))
        diag = T[jacobi_to_ultraspherical(k-1, jacobi_α(src)) for k in 1:length(src)]
        ArrayOperator(Diagonal(diag), src, dest)
    else
        generic_conversion_using_recurrences(T, src, dest; options...)
    end
end

function conversion_using_recurrences(::Type{T}, src::Ultraspherical, dest::Jacobi; options...) where T
    if (jacobi_α(dest) == jacobi_β(dest)) && (jacobi_α(src) ≈ jacobi_α(dest))
        inv(conversion_using_recurrences(T, dest, src); options...)
    else
        generic_conversion_using_recurrences(T, src, dest; options...)
    end
end


# Conversion from one ultraspherical basis to the ultraspherical basis with lambda+1
# Implementation of formula (3.3) of Olver and Townsend, A fast and well-conditioned spectral method,
# SIAM Review, 2013.
function banded_ultraspherical_to_ultraspherical(::Type{T}, n, λ) where T
    A = BandedMatrix{T}(undef, (n,n), (0,2))
    if n > 0
        A[1,1] = 1
    end
    if n > 1
        A[2,2] = λ / (λ+1)
        A[1,2] = 0
    end
    for k in 2:n-1
        K = k+1
        A[K,K] = λ / (λ+k)
        A[K-1,K] = 0
        A[K-2,K] = -λ / (λ+k)
    end
    A
end

# Convert from ChebyshevT expansion to ultraspherical expansion with lambda=1
function banded_chebyshevt_to_ultraspherical(::Type{T}, n) where T
    A = BandedMatrix{T}(undef, (n,n), (0,2))
    if n > 0
        A[1,1] = 1
    end
    onehalf = one(T)/2
    if n > 1
        A[1,2] = 0
        A[2,2] = onehalf
    end
    for k in 2:n-1
        A[k+1,k+1] = onehalf
        A[k-1,k+1] = -onehalf
        A[k,k+1] = 0
    end
    A
end

function conversion_using_recurrences(::Type{T}, src::Ultraspherical, dest::Ultraspherical; options...) where T
    @assert length(src) == length(dest)
    if ultraspherical_λ(src) ≈ ultraspherical_λ(dest)-1
        ArrayOperator(banded_ultraspherical_to_ultraspherical(T, length(src), ultraspherical_λ(src)), src, dest)
    else
        generic_conversion_using_recurrences(T, src, dest; options...)
    end
end

function conversion_using_recurrences(::Type{T}, src::ChebyshevT, dest::Ultraspherical; options...) where T
    @assert length(src) == length(dest)
    if ultraspherical_λ(dest) == 1
        ArrayOperator(banded_chebyshevt_to_ultraspherical(T, length(src)), src, dest)
    else
        generic_conversion_using_recurrences(T, src, dest; options...)
    end
end

function diagonal_chebyshevt_to_jacobi(::Type{T}, n::Int) where T
    A = T[chebyshevt_to_jacobi(k, T) for k in 0:n-1]
    Diagonal(A)
end

function conversion_using_recurrences(::Type{T}, src::ChebyshevT, dest::Jacobi; options...) where T
    @assert length(src) == length(dest)
    onehalf = one(T)/2
    if jacobi_α(dest) ≈ -onehalf && jacobi_β(dest) ≈ -onehalf
        ArrayOperator(diagonal_chebyshevt_to_jacobi(T, length(src)), src, dest)
    else
        generic_conversion_using_recurrences(T, src, dest; options...)
    end
end

function conversion_using_recurrences(::Type{T}, src::Jacobi, dest::ChebyshevT; options...) where T
    onehalf = one(T)/2
    if jacobi_α(src) ≈ -onehalf && jacobi_β(src) ≈ -onehalf
        inv(conversion_using_recurrences(T, dest, src; options...))
    else
        generic_conversion_using_recurrences(T, src, dest; options...)
    end
end

function diagonal_chebyshevu_to_jacobi(::Type{T}, n::Int) where T
    A = T[chebyshevu_to_jacobi(k, T) for k in 0:n-1]
    Diagonal(A)
end

function conversion_using_recurrences(::Type{T}, src::ChebyshevU, dest::Jacobi; options...) where T
    @assert length(src) == length(dest)
    onehalf = one(T)/2
    if jacobi_α(dest) ≈ onehalf && jacobi_β(dest) ≈ onehalf
        ArrayOperator(diagonal_chebyshevu_to_jacobi(T, length(src)), src, dest)
    else
        generic_conversion_using_recurrences(T, src, dest; options...)
    end
end

function conversion_using_recurrences(::Type{T}, src::Jacobi, dest::ChebyshevU; options...) where T
    onehalf = one(T)/2
    if jacobi_α(src) ≈ onehalf && jacobi_β(src) ≈ onehalf
        inv(conversion_using_recurrences(T, dest, src; options...))
    else
        generic_conversion_using_recurrences(T, src, dest; options...)
    end
end
