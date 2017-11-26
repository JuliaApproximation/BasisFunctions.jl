# ortho_util.jl
# ortho_util.jl

"""
    Stieltjes Algorithm

    Given nodes (t_k) and weights (w_k) of a quadrature rule, the Stieltjes Algorithm
    calculates the N first recurrence coefficients of the monic polynomials (p_k) satisfying

    '''
        p_{k+1}(t) = (t-a_k)p_k(t)-b_kp_{k-1}(t)
        p_0(t) = 1, p_{-1}(t) = 0
    '''

    and orthogonal with respect to the inner product induced by the quadrature

    '''
        \sum w_k p_m(t_k)q_n(t_k) = \delta_{m,n}
    '''

    See Gautschi
"""
function stieltjes(N, x::Array{T},w::Array{T}) where {T}
    # ####
    # Remove data with zero weights done in Gautschi
    # ####
    # indices = sortperm(w)
    # x = x[indices]
    # w = w[indices]
    # index = minimum(find(!(w==0)))
    # w = w[index:length(w)]
    # x = x[index:length(w)]
    #
    # indices = sortperm(x)
    # x = x[indices]
    # w = w[indices]

    Ncap=length(x)
    # ####
    if N < 1 || N > Ncap
        error("Too little quadrature points")
    end

    a = zeros(N)
    b = zeros(N)

    p1 = zeros(Ncap,1)
    p2 = ones(Ncap,1)
    # The following scaling has to be adopted in case the
    # orthogonal polynomials underflow or overflow. In the
    # former case, c has to be chosen sufficiently large
    # positive, in the latter case sufficiently small
    # positive. (The example below relates to Table2_9.m.)
    #
    # if N==320
    #     c=1e50;
    #     p2=c*p2;
    #     s0=c^2*s0;
    # end
    p0 = similar(p1)
    scratch = similar(p1)
    stieltjes!(a,b,N,x,w,p0,p1,p2,scratch)
    a,b
end

"""
    In-place Stieltjes Algorithm

    see `stieltjes`

    `p0`, `p1`, `p2`, and `scratch` are vectors of size `N`
"""
function stieltjes!(a::Array{T},b::Array{T},N,x::Array{T},w::Array{T},p0::Array{T},p1::Array{T},p2::Array{T},scratch::Array{T}) where {T}
    tiny = 10*typemin(T)
    huge = .1*typemax(T)

    s0 = sum(w)
    a[1] = dot(x,w)/s0
    b[1] = s0
    for k=1:N-1
        copy!(p0,p1)
        copy!(p1,p2)
        p2 .= (x.-a[k]).*p1.-b[k].*p0
        scratch .= p2.^2
        s1 = dot(w,scratch)
        scratch .= w.*p2.^2
        s2 = dot(x,scratch)
        scratch .= abs.(p2)
        if (maximum(scratch)>huge || abs(s2)>huge)
            error("impending overflow in stieltjes for k=$(k)")
        end
        if abs(s1)<tiny
            error("impending underflow in stieltjes for k=$(k)")
        end
        a[k+1] = s2/s1
        b[k+1] = s1/s0
        s0 = s1
    end
end

"""
    Modified Chebyshev Algorithm

    Implements the map of the modified moments

    '''
        \mu_k = \int p_k(x) dλ(x), k=0,...,2N-1
    '''
    where (p_k) are some monic orthogonal polynomials satisfying the recurrence relation
    '''
        p_{k+1}(t) = a_k(t_k)p_k(t) - b_kp{k-1}(t)
        p_0(t) = 1, p_{-1}(t) = 0
    '''
    to the first recurrence coefficients (α_k), (β_k), with k=0,...,N-1, such that
    the monic polynomials (q_k) satisfying
    '''
        q_{k+1}(t) = α_k(t_k)q_k(t) - β_kq{k-1}(t)
        q_0(t) = 1, q_{-1}(t) = 0
    '''
    are orthogonal with respect to the measure λ.

    By default a, and b are zero vectors and therefore the moments are assumed to be
    '''
        \mu_k = \int x^k dλ(x), k=0,...,2N-1
    '''
    This procedure is however not stable.

    Gautschi Algorithm 2.1, p77
"""
function modified_chebyshev(m::Array{T}, a=zeros(T,length(m)), b=zeros(T,length(m))) where {T}
    L = length(m)
    n = L>>1
    α = Array{T}(n)
    β = Array{T}(n)
    σmone = Array{T}(L)
    σzero = Array{T}(L)
    σ = Array{T}(L)

    @time modified_chebyshev_algorithm!(α,β,m,a,b,σ,σzero,σmone,n,1)
    α, β
end

"""
    Modified Chebyshev Algorithm

    see `chebyshev`

    `σ`, `σzero`, and `σmone` are vectors of size or larger then `2n`
    `n` is the number of coefficients required and `os` is the offset, i.e.,
    the initial index that can be used in `σ`, `σzero`, and `σmone`.
"""
function modified_chebyshev!(α,β,m,a,b,σ,σzero,σmone,n=length(α),os=1)
    # Initialisation
    α[1] = a[1]+m[2]/m[1]
    β[1] = m[1]
    fill!(σmone,0)
    copy!(σzero,os,m,os,2n)
    # continue
    for k=1:n-1
        for l in k+os:2n-k-1+os
            σ[l] = σzero[l+1]-(α[k]-a[l]).*σzero[l]-β[k].*σmone[l]+b[l].*σzero[l-1]
        end
        α[k+1] = a[k+1]
        α[k+1] += σ[k+2]/σ[k+1]
        α[k+1] = a[k+1]+σ[k+2]/σ[k+1]-σzero[k+1]/σzero[k]
        β[k+1] = σ[k+1]/σzero[k]
        copy!(σmone,os,σzero,os,2n)
        copy!(σzero,os,σ,os,2n)
    end
    nothing
end

"""
    Transforms the recurrence coefficients (a_k), (b_k), and (c_k) of orthonormal
    orthogonal polynomials (q_k), which satisfy the three-term recurrence relation
    '''
        q_{k+1}(t) = (a_kt+b_k)q_k(t)-c_kq_{k-1}(t)
    '''
    to the recurrence coefficients (α_k) and (β_k) the associated
    monic polynomials (p_k), such that
    '''
        p_{k+1}(t) = (t-α_k)p_k(t)-β_kp_{k-1}(t)
    '''

    `b0` is the first element of β and should be equal to the first moment of the
    measure the polynomials are orthogonal to.
"""
function monic_recurrence_coefficients(a::Array{T},b::Array{T},c::Array{T},b0=0) where {T}
    n = length(a)
    α = Array{T}(n)
    β = Array{T}(n)
    monic_recurrence_coefficients!(α,β,a,b,c,b0)
    α, β
end

"""
    See `monic_recurrence_coefficients`
"""
function monic_recurrence_coefficients!(α,β,a,b,c,b0)
    n = length(α)
    # We keep track of the leading order coefficients of p_{k-1}, p_k and
    # p_{k+1} in γ_km1, γ_k and γ_kp1 respectively. The recurrence relation
    # becomes, with p_k(x) = γ_k q_k(x):
    #
    # γ_{k+1} q_{k+1}(x) = (A_k x + B_k) γ_k q_k(x) - C_k γ_{k-1} q_{k-1}(x).
    #
    # From this we derive the formulas below.

    # We start with k = 1 and recall that p_0(x) = 1 and p_1(x) = A_0 x + B_0,
    # hence γ_0 = 1 and γ_1 = A_0.
    γ_km1 = 1
    # γ_k = rec_An(b, 0)
    γ_k = a[1]
    # α[1] = -rec_Bn(b, 0) / rec_An(b, 0)
    α[1] = -b[1]/a[1]
    # We arbitrarily choose β_0 = β[1] = 0. TODO: change later, should be first moment
    β[1] = b0

    # We can now loop for k from 1 to n.
    for k in 1:n-1
        γ_kp1 = a[k+1] * γ_k
        α[k+1] = -b[k+1] * γ_k / γ_kp1
        β[k+1] =  c[k+1] * γ_km1 / γ_kp1
        γ_km1 = γ_k
        γ_k = γ_kp1
    end
    α, β
end

"""
    Transforms the `N` recurrence coefficients (α_k) and (β_k) of monic
    orthogonal polynomials (p_k), which satisfy the three-term recurrence relation
    '''
        p_{k+1}(t) = (t-α_k)p_k(t)-β_kp_{k-1}(t)
    '''
    to the `N-1` first recurrence coefficients (a_k), (b_k), and (c_k) the associated
    monic polynomials (q_k), such that

    '''
        q_{k+1}(t) = (a_kt+b_k)q_k(t)-c_kq_{k-1}(t)
    '''
"""
function monic_to_orthonormal_recurrence_coefficients(α::Array{T}, β::Array{T}) where {T}
    n = length(α)
    a = Array{T}(n-1)
    b = Array{T}(n-1)
    c = Array{T}(n-1)
    monic_to_orthonormal_recurrence_coefficients!(a,b,c,α,β)
    a,b,c
end

"""
    See `monic_to_orthonormal_recurrence_coefficients`
"""
function monic_to_orthonormal_recurrence_coefficients!(a,b,c,α,β)
    a .= 1./sqrt.(β[2:end])
    b .= -α[1:end-1]./sqrt.(β[2:end])
    c .= sqrt.(β[1:end-1]./β[2:end])
    a,b,c
end
