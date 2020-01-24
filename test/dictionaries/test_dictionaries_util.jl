
# Laguerre and Hermite fail due to linear algebra problems in BigFloat
supports_approximation(::Laguerre{BigFloat}) = false
supports_approximation(::Hermite{BigFloat}) = false
# It is difficult to do approximation in subsets and operated sets generically
supports_approximation(::Subdictionary) = false
supports_approximation(::OperatedDict) = false
supports_approximation(dict::TensorProductDict) =
    reduce(&, map(supports_approximation, elements(dict)))
# Monomials and rationals have no associated domain
supports_approximation(::Monomials) = false
supports_approximation(::Rationals) = false

# Make a simple periodic function for Fourier and other periodic sets
suitable_function(::Fourier) =  x -> 1/(10+cos(2*pi*x))
# The function has to be periodic and even symmetric
suitable_function(::CosineSeries) =  x -> 1/(10+cos(2*pi*x))
# The function has to be periodic and odd symmetric
suitable_function(::SineSeries) =  x -> x^3*(1-x)^3
# We use a function that is smooth and decays towards infinity
suitable_function(::Laguerre) = x -> 1/(1000+(2x)^2)
suitable_function(::Hermite) = x -> 1/(1000+(2x)^2)
suitable_function(dict::OperatedDict) = suitable_function(src(dict))
# Make a tensor product of suitable functions
function suitable_function(dict::TensorProductDict)
    if dimension(dict) == 2
        f1 = suitable_function(element(dict,1))
        f2 = suitable_function(element(dict,2))
        (x,y) -> f1(x)*f2(y)
    elseif dimension(dict) == 3
        f1 = suitable_function(element(dict,1))
        f2 = suitable_function(element(dict,2))
        f3 = suitable_function(element(dict,3))
        (x,y,z) -> f1(x)*f2(y)*f3(z)
    end
    # We should never get here
end
# Make a suitable function by undoing the map
function suitable_function(dict::MappedDict)
    f = suitable_function(superdict(dict))
    m = inv(mapping(dict))
    x -> f(m*x)
end
function suitable_function(dict::WeightedDict1d)
    f = suitable_function(superdict(dict))
    g = weightfunction(dict)
    x -> g(x) * f(x)
end
function suitable_function(dict::WeightedDict2d)
    f = suitable_function(superdict(dict))
    g = weightfunction(dict)
    (x,y) -> g(x, y) * f(x, y)
end

suitable_interpolation_grid(basis::TensorProductDict) =
    ProductGrid(map(suitable_interpolation_grid, elements(basis))...)
suitable_interpolation_grid(basis::SineSeries) = MidpointEquispacedGrid{domaintype(basis)}(length(basis), 0, 1)
suitable_interpolation_grid(basis::WeightedDict) = suitable_interpolation_grid(superdict(basis))
suitable_interpolation_grid(basis::OperatedDict) = suitable_interpolation_grid(src(basis))
