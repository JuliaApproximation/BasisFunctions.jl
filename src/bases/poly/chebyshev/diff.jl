##################
# Differentiation
##################

function unsafe_eval_element_derivative(b::ChebyshevT{T}, idx::PolynomialDegree, x, order) where {T}
	@assert order >= 0
	if order == 0
		unsafe_eval_element(b, idx, x)
	elseif order == 1
	    d = degree(idx)
	    if d == 0
	        zero(T)
	    else
	        d * unsafe_eval_element(ChebyshevU{T}(length(b)), idx-1, x)
	    end
	else
		error("Higher order derivatives of Chebyshev polynomials not implemented.")
	end
end

# Chebyshev differentiation is so common that we make it its own type
struct ChebyshevDifferentiation{T} <: DictionaryOperator{T}
	src		::	Dictionary
	dest	::	Dictionary
    order   ::  Int
end

ChebyshevDifferentiation(src::Dictionary, dest::Dictionary, order::Int = 1) =
	ChebyshevDifferentiation{operatoreltype(src,dest)}(src, dest, order)

ChebyshevDifferentiation(src::Dictionary, order::Int = 1) =
    ChebyshevDifferentiation(src, src, order)

order(op::ChebyshevDifferentiation) = op.order

string(op::ChebyshevDifferentiation) = "Chebyshev differentiation matrix of order $(order(op)) and size $(length(src(op)))"

similar_operator(op::ChebyshevDifferentiation, src, dest) =
    ChebyshevDifferentiation(src, dest, order(op))

unsafe_wrap_operator(src, dest, op::ChebyshevDifferentiation) = similar_operator(op, src, dest)

function adjoint(op::ChebyshevDifferentiation)
    @warn "Inefficient adjoint of `ChebyshevDifferentiation`"
    ArrayOperator(adjoint(Matrix(op)), dest(op), src(op))
end

conj(op::ChebyshevDifferentiation) = op


# TODO: this allocates lots of memory...
function apply!(op::ChebyshevDifferentiation, coef_dest, coef_src)
    #	@assert period(dest)==period(src)
    n = length(coef_src)
    T = eltype(coef_src)
    tempc = coef_src[:]
    tempr = coef_src[:]
    for o = 1:order(op)
        tempr = zeros(T,n)
        # 'even' summation
        s = 0
        for i=(n-1):-2:2
            s = s+2*i*tempc[i+1]
            tempr[i] = s
        end
        # 'uneven' summation
        s = 0
        for i=(n-2):-2:2
            s = s+2*i*tempc[i+1]
            tempr[i] = s
        end
        # first row
        s = 0
        for i=2:2:n
            s = s+(i-1)*tempc[i]
        end
        tempr[1]=s
        tempc = tempr
    end
    coef_dest[1:n-order(op)] = tempr[1:n-order(op)]
    coef_dest
end

struct ChebyshevAntidifferentiation{T} <: DictionaryOperator{T}
	src		::	Dictionary
	dest	::	Dictionary
    order   ::  Int
end

ChebyshevAntidifferentiation(src::Dictionary, dest::Dictionary, order::Int = 1) =
	ChebyshevAntidifferentiation{operatoreltype(src,dest)}(src, dest, order)

ChebyshevAntidifferentiation(src::Dictionary, order::Int = 1) =
    ChebyshevAntidifferentiation(src, src, order)

order(op::ChebyshevAntidifferentiation) = op.order

string(op::ChebyshevAntidifferentiation) = "Chebyshev antidifferentiation matrix of order $(order(op)) and size $(length(src(op)))"

similar_operator(op::ChebyshevAntidifferentiation, src, dest) =
    ChebyshevAntidifferentiation(src, dest, order(op))

unsafe_wrap_operator(src, dest, op::ChebyshevAntidifferentiation) = similar_operator(op, src, dest)

# TODO: this allocates lots of memory...
function apply!(op::ChebyshevAntidifferentiation, coef_dest, coef_src)
    #	@assert period(dest)==period(src)
    T = eltype(coef_src)
    tempc = zeros(T,length(coef_dest))
    tempc[1:length(coef_src)] = coef_src[1:length(coef_src)]
    tempr = zeros(T,length(coef_dest))
    tempr[1:length(coef_src)] = coef_src[1:length(coef_src)]
    for o = 1:order(op)
        n = length(coef_src)+o
        tempr = zeros(T,n)
        tempr[2]+=tempc[1]
        tempr[3]=tempc[2]/4
        tempr[1]+=tempc[2]/4
        for i = 3:n-1
            tempr[i-1]-=tempc[i]/(2*(i-2))
            tempr[i+1]+=tempc[i]/(2*(i))
            tempr[1]+=real(-1im^i)*tempc[i]*(2*i-2)/(2*i*(i-2))
        end
        tempc = tempr
    end
    coef_dest[:]=tempr[:]
    coef_dest
end

function differentiation(::Type{T}, src::ChebyshevT, dest::ChebyshevT, order::Int; options...) where {T}
	@assert order >= 0
	if order == 0
		IdentityOperator{T}(src, dest)
	else
		ChebyshevDifferentiation{T}(src, dest, order)
	end
end

function antidifferentiation(::Type{T}, src::ChebyshevT, dest::ChebyshevT, order::Int; options...) where {T}
	@assert order >= 0
	if order == 0
		IdentityOperator{T}(src, dest)
	else
    	ChebyshevAntidifferentiation{T}(src, dest, order)
	end
end
