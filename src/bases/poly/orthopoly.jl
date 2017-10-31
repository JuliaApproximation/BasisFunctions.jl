# orthopoly.jl

"OrthogonalPolynomials is the abstract supertype of all univariate orthogonal polynomials."
abstract type OrthogonalPolynomials{T} <: PolynomialBasis{T}
end

const OPS{T} = OrthogonalPolynomials{T}

const OPSpan{A,F <: OrthogonalPolynomials} = Span{A,F}

is_orthogonal(b::OPS) = true
is_biorthogonal(b::OPS) = true

approx_length(b::OPS, n::Int) = n

derivative_space(s::OPSpan, order::Int; options...) = resize(s, length(s)-order)
antiderivative_space(s::OPSpan, order::Int; options...) = resize(s, length(s)+order)

length(o::OrthogonalPolynomials) = o.n

function dot(s::OPSpan, f1::Function, f2::Function, nodes::Array=native_nodes(set(s)); options...)
    T = coeftype(s)
	# To avoid difficult points at the ends of the domain.
	dot(x->weight(set(s),x)*f1(x)*f2(x), clip_and_cut(nodes, -T(1)+eps(real(T)), +T(1)-eps(real(T))); options...)
end

clip(a::Real, low::Real, up::Real) = min(max(low, a), up)

function clip_and_cut(a::Array{T,1}, low, up) where {T <: Real}
	clipped = clip.(a,low, up)
	t = clipped[1]
	s = 1
	for i in 2:length(a)
		t != clipped[i] && break
		t = clipped[i]
		s += 1
	end
	t = clipped[end]
	e = length(a)
	for i in length(a)-1:-1:1
		t != clipped[i] && break
		t = clipped[i]
		e -= 1
	end
	clipped[s:e]
end

has_extension(b::OPS) = true

# CAVE: we have to add F <: OrthogonalPolynomials at the end, otherwise
# OPSpan{A,F} also seems to match non-polynomial sets F (in Julia 0.6).
# Using OPSpan as types of the arguments, i.e. without parameters, is fine and
# only matches with polynomial sets. But here we use parameters to enforce that
# the two spaces have the same type of set, and same type of coefficients.
function extension_operator(s1::OPSpan{A,F}, s2::OPSpan{A,F}; options...) where {A,F <: OrthogonalPolynomials}
    @assert length(s2) >= length(s1)
    IndexExtensionOperator(s1, s2, 1:length(s1))
end

function restriction_operator(s1::OPSpan{A,F}, s2::OPSpan{A,F}; options...) where {A,F <: OrthogonalPolynomials}
    @assert length(s2) <= length(s1)
    IndexRestrictionOperator(s1, s2, 1:length(s2))
end




# Evaluate an orthogonal polynomial using the three term recurrence relation.
# The recurrence relation is assumed to have the form
#
#    p_{n+1}(x) = (A_n x - B_n) * p_n(x) - C_n * p_{n-1}(x)
#
# with the coefficients implemented by the rec_An, rec_Bn and rec_Cn functions.
function recurrence_eval(b::OPS, idx::Int, x)
	T = rangetype(b)
    z0 = one(T)
    z1 = convert(T, rec_An(b, 0) * x + rec_Bn(b, 0))

    if idx == 1
        return z0
    end
    if idx == 2
        return z1
    end

    z = z1
    for i = 1:idx-2
        z = (rec_An(b, i)*x + rec_Bn(b, i)) * z1 - rec_Cn(b, i) * z0
        z0 = z1
        z1 = z
    end
    z
end


# Default evaluation of an orthogonal polynomial: invoke the recurrence relation
eval_element(b::OPS, idx::Int, x) = recurrence_eval(b, idx, x)

eval_element_derivative(b::OPS, idx::Int, x) = recurrence_eval_derivative(b, idx, x)


function recurrence_eval_derivative(b::OPS, idx::Int, x)
	T = rangetype(b)
    z0 = one(T)
    z1 = convert(T, rec_An(b, 0) * x + rec_Bn(b, 0))
    z0_d = zero(T)
    z1_d = convert(T, rec_An(b, 0))

    if idx == 1
        return z0_d
    end
    if idx == 2
        return z1_d
    end

    z = z1
    z_d = z1_d
    for i = 1:idx-2
        z = (rec_An(b, i)*x + rec_Bn(b, i)) * z1 - rec_Cn(b, i) * z0
        z_d = (rec_An(b, i)*x + rec_Bn(b, i)) * z1_d + rec_An(b, i)*z1 - rec_Cn(b, i) * z0_d
        z0 = z1
        z1 = z
        z0_d = z1_d
        z1_d = z_d
    end
    z_d
end
