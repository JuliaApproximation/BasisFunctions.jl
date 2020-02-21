export PiecewiseConstants
struct PiecewiseConstants{S,T} <: Dictionary1d{S,T}
    n       ::  Int
end

isorthogonal(h::PiecewiseConstants, measure::FourierMeasure) = true
function isorthogonal(h::PiecewiseConstants, μ::DiscreteMeasure)
    for (i,xi) in zip(ordering(h),grid(μ))
        if xi ∉ support(h, i)
            return false
        end
    end
    return true
end
support(h::PiecewiseConstants) =
    Interval{:closed,:open,domaintype(h)}(0,1)

PiecewiseConstants(n = 10) = PiecewiseConstants{Float64}(n)
PiecewiseConstants{S}(n) where {S} = PiecewiseConstants{S,S}(n)
size(h::PiecewiseConstants) = (h.n,)

# By implementing `similar`, we support promotion of the domain type, as well
# as resizing the dictionary.
# If resizing is not possible, one can add a statement such as
# "@assert n == length(dict)", which will cause an error upon any user attempt to resize.
# Same for promotion.
similar(::PiecewiseConstants, ::Type{S}, n::Int) where {S} = PiecewiseConstants{S}(n)

support(h::PiecewiseConstants, idx::Int) =
    Interval{:closed,:open,domaintype(h)}(0,one(domaintype(h))/h.n) + (one(domaintype(h))/h.n*(idx-1))

unsafe_eval_element(h::PiecewiseConstants, idx::Int, x) = one(codomaintype(h))

interpolation_grid(h::PiecewiseConstants) = MidpointEquispacedGrid(length(h), support(h))
