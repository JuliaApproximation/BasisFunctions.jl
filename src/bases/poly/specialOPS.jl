# specialOPS.jl

"""
Creates the (normalized) half range Chebyshev polynomials of the first kind.

These orthonormal polynomials `(p_k(t))` on [-1,1] are orthogonal with respect to the weight
```
w(t) = [(1-t)(t-m(T))]^{-1/2}
```
with
```
m(T) = 1-2cosec^2(pi/(2T))
```

They appear in the context of Fourier extensions. There, an even function `f(x)` on
`[-1,1]` is approximated with a cosine series in `x` on `[-T,T]`. By the cosine mapping
```
    y = m(x) = cos(pi/T x)
```
this problem is transformed. The Fourier Extension problem on `f` is then equivalent with
the approximation of `f(m^{-1}(y))` with orthogonal polynomials `(q_k(y))` on `[cos(pi/T),1]`

By another mapping the interval `[cos(pi/T),1]` is mapped to `[-1,1]` and the polynomials `(q_k(y))` are mapped
to the polynomials `(p_k(t))`

See Huybrechs 2010, On the Fourier Extension of nonperiodic functions
    Adcock, Huybrechs, Vaquero 2014, On the numerical stability of Fourier Extension

The `indicator_function_nodes` refer to the boundary points of the domain of the function `f(x)` is approximated on.
It is assumed that `indicator_function_nodes` has the form
```
    [-1,b_1,a_2,b_2, ..., a_n, 1 ] with -1<b_1<a_2<b_2<...<a_n<1.
```
By default it is the vector [-1.,1.].

See also `HalfRangeChebyshevIkind`
"""
HalfRangeChebyshevIkind(n::Int, T::ELT, indicator_function_nodes::Vector{ELT}=default_indicator_nodes(ELT); options...) where ELT =
    HalfRangeChebyshev(n, ELT(-1//2), T, indicator_function_nodes; options...)

"""
Creates the (normalized) half range Chebyshev polynomials of the second kind.

These orthonormal polynomials `(p_k(t))` on [-1,1] are orthogonal with respect to the weight
```
w(t) = [(1-t)(t-m(T))]^{1/2}
```
with
```
m(T) = 1-2cosec^2(pi/(2T))
```

They appear in the context of Fourier extensions. There, an odd function `f(x)` on
`[-1,1]` is approximated with a sine series in `x` on `[-T,T]`.

See also `HalfRangeChebyshevIkind`
"""
HalfRangeChebyshevIIkind(n::Int, T::ELT, indicator_function_nodes::Vector{ELT}=default_indicator_nodes(ELT); options...) where ELT =
    HalfRangeChebyshev(n, ELT(1//2), T, indicator_function_nodes; options...)

m_forward = (x,T)->2*(cos(pi/T*x)-cos(pi/T))/(1-cos(pi/T))-1
m_inv = (t,T)->T/pi*acos(cos(pi/T)+(1-cos(pi/T))/2*(t+1))
weight_of_indicator(T,indicator) = t->.5*(indicator(-m_inv(t,T))+indicator(m_inv(t,T)))
default_indicator_nodes(ELT) = [-ELT(1),ELT(1)]

function HalfRangeChebyshev(n::Int, α, T::ELT, indicator_function_nodes::Vector{ELT}; options...) where ELT
    my_quadrature_rule = n->_halfrangechebyshevweights(n, α, T, indicator_function_nodes)
    OrthonormalOPSfromQuadrature(n, my_quadrature_rule, interval(-one(ELT), one(ELT)); options...)
end

function _halfrangechebyshevweights(n, α::ELT, T::ELT, indicator_function_nodes::Vector{ELT}) where {ELT}
    @assert indicator_function_nodes[1] == -1 && indicator_function_nodes[end] == 1
    @assert reduce(&, true, indicator_function_nodes[1:end-1] .< indicator_function_nodes[2:end])
    @assert iseven(length(indicator_function_nodes))

    if α < 0
        C = 2T/pi
    else
        C = T*(1-cos(pi/T))^2/2/pi
    end
    if length(indicator_function_nodes)==2
        nodes, weights = gaussjacobi(n, α, ELT(0))
        Λ = weight_of_indicator(T,x->1)
        modified_weights = weights.*((nodes.-m_forward(T,T)).^α).*Λ.(nodes)*C
        nodes, modified_weights
    else
        indicator = indicator_function(indicator_function_nodes)
        mapped_nodes = sort(unique(map(x->m_forward(x,T), push!(indicator_function_nodes,0))))
        pop!(indicator_function_nodes)
        no_intervals = length(mapped_nodes)-1
        n_interval = n
        N = n_interval*no_intervals
        nodes = zeros(ELT,N)
        weights = zeros(ELT,N)


        for i in 1:no_intervals
            nodes_interval, weights_interval = gaussjacobi(n_interval, ELT(0), ELT(0))
            a = mapped_nodes[i]
            b = mapped_nodes[i+1]
            nodes_interval[:] .= a + (b-a)/2*(nodes_interval+1)
            weights_interval[:] .= weights_interval*(b-a)/2
            Δ = weight_of_indicator(T,indicator)
            weights_interval[:] .= C*weights_interval.*(1-nodes_interval).^α.*(nodes_interval-m_forward(T,T)).^α.*Δ.(nodes_interval)

            nodes[1+(i-1)*n_interval:i*n_interval] .= nodes_interval[:]
            weights[1+(i-1)*n_interval:i*n_interval] .= weights_interval[:]
        end
        nodes, weights
    end
end

indicator_function(nodes) = x-> reduce(|, false, nodes[1:2:end] .<= x .<= nodes[2:2:end])

function WaveOPS(n::Int,omega::ELT; options...) where {ELT}
    my_quadrature_rule = n->_wavePolynomialweight(n, omega)
    BasisFunctions.OrthonormalOPSfromQuadrature(n, my_quadrature_rule, interval(-one(ELT), one(ELT)), (x->(abs(x)<=1) ? exp(1im*omega*x) : ELT(0)); options...)
end

function _wavePolynomialweight(n, omega::ELT) where ELT
    nodes, weights = gaussjacobi(n,ELT(0),ELT(0))
    weights = weights.*exp.(1im.*omega.*nodes)
    nodes, weights
end
