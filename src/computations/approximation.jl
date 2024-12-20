
########################
# Generic approximation
########################

const ANYFUNCTION = Union{Function,Expansion,TypedFunction}

"""
The `approximation` function returns an operator that can be used to approximate
a function in the function set. This operator maps a grid to a set of coefficients.
"""
approximation(Φ; discrete = true, options...) =
  discrete ?
    discrete_approximation(Φ; options...) :
    continuous_approximation(Φ; options...)

function discrete_approximation(Φ::Dictionary; options...)
    if isbasis(Φ) && hasinterpolationgrid(Φ)
        interpolation(Φ; options...)
    else
        leastsquares(Φ; options...)
    end
end

continuous_approximation(Φ::Dictionary; options...) = inv(gram(Φ; options...))

# Automatically sample a function if an operator is applied to it
(*)(op::DictionaryOperator, f::ANYFUNCTION) = op * project(src(op), f)
Base.:\(op::DictionaryOperator, f::ANYFUNCTION) = op \ project(dest(op), f)

project(Φ::GridBasis, f::ANYFUNCTION; options...) = sample(Φ, f)

approximate(Φ::Dictionary, f::Function; options...) =
    Expansion(Φ, approximation(Φ; options...) * f)



#################
# Interpolation
#################

function interpolation_matrix(Φ::Dictionary, pts, T=coefficienttype(Φ))
    @assert length(Φ) == length(pts)
    evaluation_matrix(Φ, pts, T)
end

interpolation(Φ::Dictionary, args...; options...) =
    interpolation(operatoreltype(Φ), Φ, args...; options...)

interpolation(::Type{T}, Φ::Dictionary; options...) where {T} =
    interpolation(T, Φ, interpolation_grid(Φ); options...)

interpolation(::Type{T}, Φ::Dictionary, grid::AbstractArray; options...) where {T} =
    interpolation(T, Φ, GridBasis{T}(grid); options...)

function interpolation(::Type{T}, Φ::Dictionary, gb::GridBasis; options...) where {T}
    if hasinterpolationgrid(Φ) && interpolation_grid(Φ) == grid(gb) && hastransform(Φ, gb)
        transform(T, gb, Φ; options...)
    else
        default_interpolation(Φ, gb; options...)
    end
end

default_interpolation(Φ::Dictionary, gb::GridBasis; options...) =
    default_interpolation(operatoreltype(Φ, gb), Φ, gb; options...)

default_interpolation(::Type{T}, Φ::Dictionary, gb::GridBasis; options...) where {T} =
    QR_solver(evaluation(T, Φ, grid(gb)))

interpolate(Φ::Dictionary, pts, f, T=coefficienttype(Φ)) = Expansion(Φ,interpolation(T, Φ, pts) * f)

"""
Compute the weights of an interpolatory quadrature rule.
"""
function quadrature_weights(Φ::Dictionary, pts)
    A = interpolation_matrix(Φ, pts)
    C = inv(A)
    [integral(Expansion(Φ, C[:,k])) for k in 1:length(Φ)]
end

########################
# Generic least squares
########################


function leastsquares_matrix(Φ::Dictionary, pts, T=coefficienttype(Φ))
    @assert length(Φ) <= length(pts)
    evaluation_matrix(Φ, pts, T)
end

leastsquares(Φ::Dictionary, args...; options...) =
    leastsquares(operatoreltype(Φ), Φ, args...; options...)

function leastsquares(::Type{T}, Φ::Dictionary; samplingfactor = 2, options...) where {T}
    if hasinterpolationgrid(Φ)
        Φ2 = resize(Φ, samplingfactor*length(Φ))
        ls_grid = interpolation_grid(Φ2)
    else
        ls_grid = EquispacedGrid(samplingfactor*length(Φ), support(Φ))
    end
    leastsquares(T, Φ, ls_grid; options...)
end

leastsquares(::Type{T}, Φ::Dictionary, grid::AbstractArray; options...) where {T} =
    leastsquares(T, Φ, GridBasis{T}(grid); options...)

function leastsquares(T, Φ::Dictionary, gb::GridBasis; options...)
    if hasinterpolationgrid(Φ)
        larger_Φ = resize(Φ, size(gb))
        if hastransform(larger_Φ, gb)
            R = restriction(T, larger_Φ, Φ; options...)
            E = transform(T, gb, larger_Φ; options...)
            R * E
        else
            default_leastsquares(Φ, gb, T; options...)
        end
    else
        default_leastsquares(Φ, gb, T; options...)
    end
end

default_leastsquares(Φ::Dictionary, gb::GridBasis, T=operatoreltype(Φ,gb); options...) =
    QR_solver(ArrayOperator(leastsquares_matrix(Φ, grid(gb), T), Φ, gb))


########################
# Inverse of a function
########################

"""
    inversefunction(F[, domain; n])

Compute an approximation to the inverse of `F`, using an appropriate basis with `n` degrees of freedom.
"""
inversefunction(F::Expansion; args...) = inversefunction(F, support(F); args...)

function inversefunction(F, domain::AbstractInterval; n = 2*length(F))
    Fd = differentiate(F)
    if length(roots(Fd; all=false)) > 0
        throw(ArgumentError("Function is not invertible on its domain."))
    end
    
    # compute the value t such that F(t)=z.
    function inversevalue(z)
        r = roots(F - z; all=false)
        if length(r) != 1
            throw(ArgumentError("No unique inverse value."))
        end
        r[1]
    end
    
    a = leftendpoint(domain)
    b = rightendpoint(domain)
    A = F(a)
    B = F(b)
    G = ChebyshevT(n) → (A..B)
    Expansion(G, approximation(G) * inversevalue)
end
