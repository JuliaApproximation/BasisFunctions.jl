if !isdefined(DomainSets, :Domain1d)
    # Convenient aliases
    const Domain1d{T <: Number} = Domain{T}
    const Domain2d{T} = EuclideanDomain{2,T}
    const Domain3d{T} = EuclideanDomain{3,T}
    const Domain4d{T} = EuclideanDomain{4,T}
    export Domain1d, Domain2d, Domain3d, Domain4d
end

iscompatible(map1::AbstractMap, map2::AbstractMap) = map1==map2

iscompatible(map1::AffineMap, map2::AffineMap) = (map1.a ≈ map2.a) && (map1.b ≈ map2.b)

iscompatible(domain1::Domain, domain2::Domain) = domain1 == domain2

# Convenience functions to make some simple domains
interval() = UnitInterval{Float64}()

circle(::Type{T} = Float64) where {T} = UnitCircle{T}()
circle(radius::Number) = radius * circle(float(typeof(radius)))
circle(radius::Number, center::AbstractVector) = circle(radius) + center

sphere(::Type{T} = Float64) where {T} = UnitSphere{T}()
sphere(radius::Number) = radius * sphere(float(typeof(radius)))
sphere(radius::Number, center::AbstractVector) = sphere(radius) + center

disk(::Type{T} = Float64) where {T} = UnitDisk{T}()
disk(radius::Number) = radius * disk(typeof(radius))
disk(radius::Number, center::AbstractVector) = disk(radius) + center

ball(::Type{T} = Float64) where {T} = UnitBall{T}()
ball(radius::Number) = radius * ball(typeof(radius))
ball(radius::Number, center::AbstractVector) = ball(radius) + center

simplex(::Type{Val{N}}, ::Type{T} = Float64) where {T,N} = UnitSimplex{N,T}()
# TODO: this will become:
# simplex(::Type{Val{N}}, ::Type{T} = Float64) where {T,N} = UnitSimplex{N,T}()

cube(::Type{Val{N}}, ::Type{T} = Float64) where {N,T} = cartesianproduct(UnitInterval{T}(), Val{N})
cube() = cube(Val{3})
rectangle(a, b, c, d) = (a..b) × (c..d)
cube(a, b, c, d, e, f) = (a..b) × (c..d) × (e..f)
# This one is not type-stable
cube(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T} = ProductDomain(map((ai,bi)->ClosedInterval{T}(ai,bi), a, b)...)
# This one isn't either
cube(a::AbstractVector{T}, b::AbstractVector{T}) where {T} = cube(tuple(a...), tuple(b...))

convert(::Type{Domain{T}}, ::UnitInterval{S}) where{T,S} = UnitInterval{T}()
