
export PiecewiseConstants

"A dictionary of piecewise constant functions on `[0,1]`."
struct PiecewiseConstants{T} <: Dictionary1d{T,T}
    n       ::  Int
end

PiecewiseConstants(n = 10) = PiecewiseConstants{Float64}(n)

## Basic interface

# size should return a tuple (length is defined automatically in terms of size)
size(h::PiecewiseConstants) = (h.n,)

# By implementing `similar`, we support promotion of the domain type, as well
# as resizing of the dictionary.
# If resizing is not possible, one can add a statement such as
# "@assert n == length(dict)", which will cause an error upon any user attempt to resize.
# Same for promotion.
similar(::PiecewiseConstants, ::Type{S}, n::Int) where {S} = PiecewiseConstants{S}(n)

# Evaluation of the basis function:
unsafe_eval_element(h::PiecewiseConstants{T}, idx::Int, x) where {T} = one(T)

# The interpolation_grid returns a set of points such that the interpolation
# problem is uniquely solvable. This is useful for function approximation.
interpolation_grid(h::PiecewiseConstants) = MidpointEquispacedGrid(length(h), support(h))

# The support of our dictionary is the half-open interval [0,1)
support(h::PiecewiseConstants{T}) where {T} = Interval{:closed,:open,T}(0,1)

# We also implement the support of each basis function. By default, this would
# be the default of the dictionary itself.
support(h::PiecewiseConstants{T}, idx::Int) where {T} =
    Interval{:closed,:open,T}(0,one(T)/h.n) .+ (one(T)/h.n*(idx-1))



## Orthogonality properties

# The measure of a dictionary is used when computing projections (inner products
# of a function with a basis function)
measure(h::PiecewiseConstants{T}) where {T} = FourierMeasure{T}()

isorthogonal(h::PiecewiseConstants, measure::FourierMeasure) = true

# Is our dictionary orthogonal with respect to a discrete measure? We check all points.
function isorthogonal(h::PiecewiseConstants, μ::DiscreteMeasure)
    for (i,xi) in zip(ordering(h),points(μ))
        if xi ∉ support(h, i)
            return false
        end
    end
    return true
end
