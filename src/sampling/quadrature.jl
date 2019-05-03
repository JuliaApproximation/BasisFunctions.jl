
function trapezoidal_rule(n::Int, a = -1.0, b = 1.0, ::Type{T} = typeof((b-a)/n)) where {T}
    grid = EquispacedGrid(n, a, b)
    weights = step(grid) * ones(T, n)
    weights[1] /= 2
    weights[end] /= 2
    grid, weights
end

function rectangular_rule(n::Int, a = 0.0, b = 1.0, ::Type{T} = typeof((b-a)/n)) where {T}
    grid = PeriodicEquispacedGrid(n, a, b)
    weight = step(grid)
    grid, weight
end

# The implementation of the first and second Fejer rule and of Clenshaw-Curtis
# quadrature is based on the exposition in:
# J. Waldvogel, Fast construction of the Fejer and Clenshaw-Curtis quadrature
# rules, BIT 43(1):1-18, 2003.
function fejer_vector1!(v)
    n = length(v)
    T = eltype(v)
    v[1] = T(2)
    for k in 1:(n-1)>>1
        r = T(2)/(1-4k^2)*exp(im*k*T(pi)/n)
        v[k+1] = r
        v[n-k+1] = conj(r)
    end
    if iseven(n)
        v[n>>1+1] = 0
    end
    v
end

function fejer_first_rule(n::Int, ::Type{T} = Float64) where {T}
    v = zeros(Complex{T}, n)
    fejer_vector1!(v)
    ChebyshevNodes{T}(n), real(ifft(v))
end

function fejer_vector2!(v)
    n = length(v)
    T = eltype(v)
    v[1] = T(2)
    for k in 1:(n-1)>>1
        r = T(2)/(1-4k^2)
        v[k+1] = r
        v[n-k+1] = r
    end
    if iseven(n)
        k = n>>1-1
        v[k+1] = T(2)/(1-4k^2)
        v[n>>1+1] = T(n-3)/(2*(n>>1)-1) - 1
    end
    v
end

# TODO: verify, is this correct?
function fejer_second_rule(n::Int, ::Type{T} = Float64) where {T}
    v = zeros(T, n)
    weights = zeros(T, n+1)
    fejer_vector2!(v)
    weights[1:n] = real(ifft(v))
    weights[n+1] = weights[1]
    ChebyshevExtremae{T}(n+1), weights
end

function cc_vector_g!(g)
    n = length(g)
    T = eltype(g)
    w0cc = one(T) / (n^2-1+mod(n,2))
    for k in 1:n
        g[k] = -w0cc
    end
    if iseven(n)
        g[n>>1+1] = w0cc * ((2-mod(n,2))*n-1)
    end
    g
end

function clenshaw_curtis(n, ::Type{T} = Float64) where {T}
    v = zeros(T, n)
    g = zeros(T, n)
    weights = zeros(T, n+1)
    fejer_vector2!(v)
    cc_vector_g!(g)
    weights[1:n] = real(ifft(v+g))
    weights[n+1] = weights[1]
    ChebyshevExtremae{T}(n+1), weights
end


function rescale_cc_quad(x, w, a, b)
    y = (reverse(x).+1)/2 * (b-a) .+ a
    v = w .* (b-a)/2
    y,v
end

function rescale_fejer_quad(x, w, a, b)
    y = (x.+1)/2 * (b-a) .+ a
    v = w .* (b-a)/2
    y,v
end

function graded_rule(sigma, a, b, M, n, T = Float64, ndiff = 2)
    xg = zeros(T, 0)
    wg = zeros(T, 0)
    q1 = one(T)
    for m = M-1:-1:1
        x,w = BasisFunctions.fejer_first_rule(n, T)
        q0 = sigma*q1
        xs,ws = rescale_fejer_quad(x, w, q0, q1)
        xg = [xg; xs]
        wg = [wg; ws]
        q1 = q0
        n -= ndiff
    end
    x,w = BasisFunctions.fejer_first_rule(n, T)
    xs,ws = rescale_fejer_quad(x, w, 0, q1)
    xg = [xg; xs]
    wg = [wg; ws]
    a .+ xg*(b-a), wg * (b-a)
end
