export Haar
struct Haar{S,T} <: Dictionary1d{S,T}
    n       ::  Int
end

isorthogonal(h::Haar, measure::FourierMeasure) = true
function isorthogonal(h::Haar, μ::DiscreteMeasure)
    for (i,xi) in zip(ordering(h),grid(μ))
        if xi ∉ support(h, i)
            return false
        end
    end
    return true
end
support(h::Haar) =
    Interval{:closed,:open,domaintype(h)}(0,1)

Haar(n = 10) = Haar{Float64}(n)
Haar{S}(n) where {S} = Haar{S,S}(n)
size(h::Haar) = (h.n,)

# By implementing `similar`, we support promotion of the domain type, as well
# as resizing the dictionary.
# If resizing is not possible, one can add a statement such as
# "@assert n == length(dict)", which will cause an error upon any user attempt to resize.
# Same for promotion.
similar(::Haar, ::Type{S}, n::Int) where {S} = Haar{S}(n)

support(h::Haar, idx::Int) =
    Interval{:closed,:open,domaintype(h)}(0,one(domaintype(h))/h.n) + (one(domaintype(h))/h.n*(idx-1))

unsafe_eval_element(h::Haar, idx::Int, x) = one(codomaintype(h))

interpolation_grid(h::Haar) = MidpointEquispacedGrid(length(h), support(h))
