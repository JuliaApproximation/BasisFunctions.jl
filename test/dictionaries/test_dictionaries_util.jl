
# Laguerre and Hermite fail due to linear algebra problems in BigFloat
supports_approximation(::Laguerre{BigFloat}) = false
supports_approximation(::Laguerre{Double64}) = false
supports_approximation(::Hermite{BigFloat}) = false
supports_approximation(::Hermite{Double64}) = false
# It is difficult to do approximation in subsets and operated sets generically
supports_approximation(::Subdictionary) = false
supports_approximation(::OperatedDict) = false
supports_approximation(dict::TensorProductDict) =
    mapreduce(supports_approximation, &, components(dict))
# Monomials and rationals have no associated domain
supports_approximation(::Monomials) = false
supports_approximation(::RationalFunctions) = false

# Make a simple periodic function for Fourier and other periodic sets
suitable_function(::BasisFunctions.FourierLike) =  x -> 1/(10+cos(2*pi*x))
suitable_function(::PeriodicSincFunctions) =  x -> cos(2*pi*x)
# The function has to be periodic and even symmetric
suitable_function(::CosineSeries) =  x -> 1/(10+cos(2*pi*x))
# The function has to be periodic and odd symmetric
suitable_function(::SineSeries) =  x -> x^3*(1-x)^3
# We use a function that is smooth and decays towards infinity
suitable_function(::Laguerre) = x -> 1/(1000+(2x)^2)
suitable_function(::Hermite) = x -> 1/(1000+(2x)^2)


suitable_interpolation_grid(basis::TensorProductDict) =
    ProductGrid(map(suitable_interpolation_grid, components(basis))...)
suitable_interpolation_grid(basis::SineSeries) = MidpointEquispacedGrid{domaintype(basis)}(length(basis), 0, 1)
suitable_interpolation_grid(basis::WeightedDict) = suitable_interpolation_grid(superdict(basis))
suitable_interpolation_grid(basis::OperatedDict) = suitable_interpolation_grid(src(basis))
