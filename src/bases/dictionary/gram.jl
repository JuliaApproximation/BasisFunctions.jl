
# Methods for the computation of Gram matrices and continuous projections in general

# By convention Gram functionality is only implemented for dictionaries that are
# associated with a measure.
hasmeasure(Φ::Dictionary) = false

# Determine a measure to use when two dictionaries are given
defaultmeasure(Φ1::Dictionary, Φ2::Dictionary) =
    _defaultmeasure(Φ1, Φ2, measure(Φ1), measure(Φ2))

function _defaultmeasure(Φ1, Φ2, m1, m2)
    if iscompatible(m1, m2)
        m1
    else
        if iscompatible(support(Φ1),support(Φ2))
            lebesguemeasure(support(Φ1))
        else
            error("Please specify which measure to use for the combination of $(Φ1) and $(Φ2).")
        end
    end
end

# Shortcut: Dictionaries of the same type have just one measure
defaultmeasure(Φ1::D, Φ2::D) where {D <: Dictionary} = measure(Φ1)


innerproduct(Φ1::Dictionary, i, Φ2::Dictionary, j; options...) =
    innerproduct(Φ1, i, Φ2, j, defaultmeasure(Φ1, Φ2); options...)

# Convert linear indices to native indices, then call innerproduct_native
innerproduct(Φ1::Dictionary, i::Int, Φ2::Dictionary, j::Int, measure; options...) =
    innerproduct_native(Φ1, native_index(Φ1, i), Φ2, native_index(Φ2, j), measure; options...)
innerproduct(Φ1::Dictionary, i, Φ2::Dictionary, j::Int, measure; options...) =
    innerproduct_native(Φ1, i, Φ2, native_index(Φ2, j), measure; options...)
innerproduct(Φ1::Dictionary, i::Int, Φ2::Dictionary, j, measure; options...) =
    innerproduct_native(Φ1, native_index(Φ1, i), Φ2, j, measure; options...)
innerproduct(Φ1::Dictionary, i, Φ2::Dictionary, j, measure; options...) =
    innerproduct_native(Φ1, i, Φ2, j, measure; options...)

# - innerproduct_native: if not specialized, called innerproduct1
innerproduct_native(Φ1::Dictionary, i, Φ2::Dictionary, j, measure; options...) =
    innerproduct1(Φ1, i,  Φ2, j, measure; options...)
# - innerproduct1: possibility to dispatch on the first dictionary without ambiguity.
#                  If not specialized, we call innerproduct2
innerproduct1(Φ1::Dictionary, i, Φ2, j, measure; options...) =
    innerproduct2(Φ1, i, Φ2, j, measure; options...)
# - innerproduct2: possibility to dispatch on the second dictionary without ambiguity
innerproduct2(Φ1, i, Φ2::Dictionary, j, measure; options...) =
    default_dict_innerproduct(Φ1, i, Φ2, j, measure; options...)


# Make a quadrature strategy using user-supplied atol and rtol if they were given
function quadstrategy(::Type{T}; atol = DomainIntegrals.default_atol(T), rtol = DomainIntegrals.default_rtol(T), maxevals=DomainIntegrals.default_maxevals() , options...) where {T}
	QuadAdaptive{T}(atol, rtol, maxevals)
end


# We make this a separate routine so that it can also be called directly, in
# order to compare to the value reported by a dictionary overriding innerproduct
function default_dict_innerproduct(Φ1::Dictionary, i, Φ2::Dictionary, j, measure::Measure; options...)
	@boundscheck checkbounds(Φ1, i)
	@boundscheck checkbounds(Φ2, j)
    domain1 = support(Φ1, i)
    domain2 = support(Φ2, j)
    domain3 = support(measure)

	# This code is delicate: it is necessary when using basis functions with compact
	# support, because numerical integration over the whole domain might miss the
	# part where the integrand is non-zero. However, the intersection of three domains
	# below might fail.
	try
    	domain = intersectdomain(domain1, domain2, domain3)
		qs = quadstrategy(prectype(Φ1, Φ2); options...)
    	unsafe_default_dict_innerproduct1(Φ1, i, Φ2, j, measure, domain1, domain2, domain3, domain, qs)
	catch e
		# Intersection or integration failed, use safe evaluation
		@warn "Error thrown, trying to recover"
		# qs = quadstrategy(prectype(Φ1, Φ2); options...)
		safe_default_dict_innerproduct(Φ1, i, Φ2, j, measure, qs)
	end
end

# Routine below is safe because it uses eval_element, and not unsafe_eval_element
safe_default_dict_innerproduct(Φ1::Dictionary, i, Φ2::Dictionary, j, measure::Measure, qs) =
	integral(qs, x->conj(eval_element(Φ1, i, x))*eval_element(Φ2, j, x), measure)

function default_dict_innerproduct(Φ1::Dictionary, i, Φ2::Dictionary, j, measure::DiscreteWeight; options...)
	@boundscheck checkbounds(Φ1, i)
	@boundscheck checkbounds(Φ2, j)
    domain1 = support(Φ1, i)
    domain2 = support(Φ2, j)
    unsafe_default_dict_innerproduct2(Φ1, i, Φ2, j, measure, domain1 ∩ domain2)
end

# unsafe for indexing
function unsafe_default_dict_innerproduct1(Φ1::Dictionary, i, Φ2::Dictionary, j, measure::Weight, d1, d2, d3, domain, qs)
    if d1 == d2 == d3
        # -> domains are the same, don't convert the measure
        unsafe_default_dict_innerproduct2(Φ1, i, Φ2, j, measure, qs)
    else
        # -> do compute on the smaller domain and convert the measure
        integral(qs, x->conj(unsafe_eval_element(Φ1, i, x))*unsafe_eval_element(Φ2, j, x)*unsafe_weightfun(measure,x), domain)
    end
end

unsafe_default_dict_innerproduct1(Φ1::Dictionary, i, Φ2::Dictionary, j, measure::Measure, d1, d2, d3, domain::IntersectDomain, qs) =
    # -> disregard the intersection domain, but use the safe eval instead to guarantee correctness
    integral(qs, x->conj(eval_element(Φ1, i, x))*eval_element(Φ2, j, x), measure)
unsafe_default_dict_innerproduct1(Φ1::Dictionary, i, Φ2::Dictionary, j, measure::Weight, d1, d2, d3, domain::IntersectDomain, qs) =
    # -> disregard the intersection domain, but use the safe eval instead to guarantee correctness
    integral(qs, x->conj(eval_element(Φ1, i, x))*eval_element(Φ2, j, x), measure)

# unsafe for indexing and for support of integral
unsafe_default_dict_innerproduct2(Φ1::Dictionary, i, Φ2::Dictionary, j, measure, qs) =
    integral(qs, x->conj(unsafe_eval_element(Φ1, i, x))*unsafe_eval_element(Φ2, j, x), measure)

function unsafe_default_dict_innerproduct2(Φ1::Dictionary, i, Φ2::Dictionary, j, measure::DiscreteWeight, domain)
	# Note: using "unsafe_eval" here is still slightly dangerous: we rely on "integral"
	# not evaluating the integrand outside the given domain for discrete weights
    integral(x->conj(unsafe_eval_element(Φ1, i, x))*unsafe_eval_element(Φ2, j, x), domain, measure)
end




gramelement(dict::Dictionary, i, j, μ = measure(dict); options...) =
    innerproduct(dict, i, dict, j, μ; options...)

# Call this routine in order to evaluate the Gram matrix entry numerically
default_gramelement(dict::Dictionary, i, j, μ = measure(dict); options...) =
    default_dict_innerproduct(dict, i, dict, j, μ; options...)


function grammatrix(Φ::Dictionary, μ = measure(Φ), T = codomaintype(Φ); options...)
    G = zeros(T, length(Φ), length(Φ))
    grammatrix!(G, Φ, μ; options...)
end

function grammatrix!(G, Φ::Dictionary, μ = measure(Φ); options...)
    n = length(Φ)
    for i in 1:n
        for j in 1:i-1
            G[i,j] = gramelement(Φ, i, j, μ; options...)
            G[j,i] = conj(G[i,j])
        end
        G[i,i] = gramelement(Φ, i, i, μ; options...)
    end
    G
end

gram(Φ::Dictionary, args...; options...) = gram(operatoreltype(Φ), Φ, args...; options...)

gram(::Type{T}, Φ::Dictionary; options...) where {T} = gram(T, Φ, measure(Φ); options...)

gram(::Type{T}, Φ::Dictionary, μ::Measure; options...) where {T} =
	gram1(T, Φ, μ; options...)

gram1(T, Φ::Dictionary, μ; options...) =
	gram2(T, Φ, μ; options...)

gram2(T, Φ, μ::Measure; options...) =
	default_gram(T, Φ, μ; options...)

gram2(T, Φ, μ::DiscreteWeight; options...) =
    gram(T, Φ, μ, points(μ), weights(μ); options...)

gram(::Type{T}, Φ, μ::DiscreteWeight, grid::AbstractGrid, weights; options...) where {T} =
    default_mixedgram_discretemeasure(T, Φ, Φ, μ, grid, weights; options...)

default_gram(Φ::Dictionary, args...; options...) =
	default_gram(operatoreltype(Φ), Φ, args...; options...)

function default_gram(::Type{T}, Φ::Dictionary, μ::Measure = measure(Φ); warnslow = true, options...) where {T}
    # warnslow && @debug "Slow computation of Gram matrix entrywise of $Φ with measure $μ)."
    A = grammatrix(Φ, μ, T; options...)
    R = ArrayOperator(A, Φ, Φ)
end

default_diagonal_gram(Φ::Dictionary, μ::Measure = measure(Φ); options...) =
	default_diagonal_gram(operatoreltype(Φ), μ; options...)

function default_diagonal_gram(::Type{T}, Φ::Dictionary, μ::Measure; options...) where {T}
    @assert isorthogonal(Φ, μ)
	n = length(Φ)
	diag = zeros(T, n)
	for i in 1:n
		diag[i] = innerproduct(Φ, i, Φ, i, μ; options...)
	end
	DiagonalOperator(Φ, diag)
end

function default_mixedgram_discretemeasure(::Type{T}, Φ1::Dictionary, Φ2::Dictionary,
			μ::DiscreteWeight, grid::AbstractGrid, weights; options...) where {T}
    E1 = evaluation(T, Φ1, grid; options...)
    E2 = evaluation(T, Φ2, grid; options...)
    W = DiagonalOperator{T}(dest(E2),dest(E1), weights)
    E1'*W*E2
end

"""
Project the function onto the space spanned by the given dictionary.
"""
project(Φ::Dictionary, f, m = measure(Φ); T = coefficienttype(Φ), options...) =
    project!(zeros(T,Φ), Φ, f, m; options...)

function project!(result, Φ, f, μ; options...)
    for i in eachindex(result)
        result[i] = innerproduct(Φ[i], f, μ; options...)
    end
    result
end



########################
# Mixed gram operators
########################


mixedgram(Φ1::Dictionary, Φ2::Dictionary, args...; options...) =
	mixedgram(operatoreltype(Φ1,Φ2), Φ1, Φ2, args...; options...)

mixedgram(::Type{T}, Φ1::Dictionary, Φ2::Dictionary; options...) where {T} =
    mixedgram(T, Φ1, Φ2, defaultmeasure(Φ1, Φ2); options...)


"""
Compute the mixed Gram matrix corresponding to two dictionaries. The matrix
has elements given by the inner products between the elements of the dictionaries,
relative to the given measure.
"""
mixedgram(::Type{T}, Φ1::Dictionary, Φ2::Dictionary, μ::Measure; options...) where {T} =
    mixedgram1(T, Φ1, Φ2, μ; options...)

# The routine mixedgram1 can be specialized by concrete subtypes of the
# first dictionary, while mixedgram2 can be specialized on the second dictionary.
# mixedgram3 can be specialized on the measure
mixedgram1(T, Φ1::Dictionary, Φ2, μ; options...) =
    mixedgram2(T, Φ1, Φ2, μ; options...)

mixedgram2(T, Φ1, Φ2::Dictionary, μ; options...) =
    mixedgram3(T, Φ1, Φ2, μ; options...)

mixedgram3(T, Φ1, Φ2, μ::Measure; options...) =
    default_mixedgram(T, Φ1, Φ2, μ; options...)

mixedgram3(T, Φ1, Φ2, μ::DiscreteWeight; options...) =
    mixedgram(T, Φ1, Φ2, μ, points(μ), weights(μ); options...)

mixedgram(T, Φ1, Φ2, μ::DiscreteWeight, grid::AbstractGrid, weights; options...) =
    default_mixedgram_discretemeasure(T, Φ1, Φ2, μ, grid, weights; options...)

function mixedgram(::Type{T}, Φ1::D, Φ2::D, μ::Measure; options...) where {T,D<:Dictionary}
    if Φ1 == Φ2
        gram(T, Φ1, μ; options...)
    else
        mixedgram1(T, Φ1, Φ2, μ; options...)
    end
end

default_mixedgram(Φ1::Dictionary, Φ2::Dictionary, args...; options...) =
	default_mixedgram(operatoreltype(Φ1, Φ2), Φ1, Φ2, args...; options...)

function default_mixedgram(::Type{T}, Φ1::Dictionary, Φ2::Dictionary, μ; warnslow = true, options...) where {T}
    # warnslow && @debug "Slow computation of mixed Gram matrix entrywise."
    A = mixedgrammatrix(Φ1, Φ2, μ, T; options...)
    ArrayOperator(A, Φ2, Φ1)
end

function mixedgrammatrix(Φ1::Dictionary, Φ2::Dictionary, μ, T = operatoreltype(Φ1,Φ2); options...)
    G = zeros(T, length(Φ1), length(Φ2))
    mixedgrammatrix!(G, Φ1, Φ2, μ; options...)
end

function mixedgrammatrix!(G, Φ1::Dictionary, Φ2::Dictionary, μ; options...)
    m = length(Φ1)
    n = length(Φ2)
    for i in 1:m
        for j in 1:n
            G[i,j] = innerproduct(Φ1, i, Φ2, j, μ; options...)
        end
    end
    G
end
