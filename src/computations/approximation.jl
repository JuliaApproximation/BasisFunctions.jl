
########################
# Generic approximation
########################

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
function (*)(op::DictionaryOperator, f::Function)
    op * project(src(op), f)
end

project(Φ::GridBasis, f::Function; options...) = sample(Φ, f)

approximate(Φ::Dictionary, f::Function; options...) =
    Expansion(Φ, approximation(Φ; options...) * f)

"""
The 2-argument approximation exists to allow you to transform any
Dictionary coefficients to any other Dictionary coefficients, including
Discrete Grid Spaces.
If a transform exists, the transform is used.
"""
function approximation(A::Dictionary, B::Dictionary, options...)
    if hastransform(A, B)
        return transform(A, B; options...)
    else
        error("Don't know a transformation from ", A, " to ", B)
    end
end


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

interpolation(::Type{T}, Φ::Dictionary, grid::AbstractGrid; options...) where {T} =
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

interpolate(Φ::Dictionary, pts, f, T=coefficienttype(s)) = Expansion(interpolation(T, Φ, pts) * f)



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

leastsquares(::Type{T}, Φ::Dictionary, grid::AbstractGrid; options...) where {T} =
    leastsquares(T, Φ, GridBasis{T}(grid); options...)

function leastsquares(T, Φ::Dictionary, gb::GridBasis; options...)
    if hasinterpolationgrid(Φ)
        larger_Φ = resize(Φ, size(gb))
        if interpolation_grid(larger_Φ) == grid(gb) && hastransform(larger_Φ, gb)
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
