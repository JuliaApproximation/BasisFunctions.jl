# test_generic_dicts.jl

# Laguerre and Hermite fail due to linear algebra problems in BigFloat
supports_approximation(s::LaguerrePolynomials{BigFloat}) = false
supports_approximation(s::HermitePolynomials{BigFloat}) = false
# It is difficult to do approximation in subsets and operated sets generically
supports_approximation(s::Subdictionary) = false
supports_approximation(s::OperatedDict) = false
supports_approximation(s::TensorProductDict) =
    reduce(&, map(supports_approximation, elements(s)))

# disable for now
supports_interpolation(s::SingletonSubdict) = false

# Make a simple periodic function for Fourier and other periodic sets
suitable_function(set::FourierBasis) =  x -> 1/(10+cos(2*pi*x))
# The function has to be periodic and even symmetric
suitable_function(set::CosineSeries) =  x -> 1/(10+cos(2*pi*x))
# The function has to be periodic and odd symmetric
suitable_function(set::SineSeries) =  x -> x^3*(1-x)^3
# We use a function that is smooth and decays towards infinity
suitable_function(set::LaguerrePolynomials) = x -> 1/(1000+(2x)^2)
suitable_function(set::HermitePolynomials) = x -> 1/(1000+(2x)^2)
suitable_function(set::OperatedDict) = suitable_function(src(set))
# Make a tensor product of suitable functions
function suitable_function(s::TensorProductDict)
    if dimension(s) == 2
        f1 = suitable_function(element(s,1))
        f2 = suitable_function(element(s,2))
        (x,y) -> f1(x)*f2(y)
    elseif dimension(s) == 3
        f1 = suitable_function(element(s,1))
        f2 = suitable_function(element(s,2))
        f3 = suitable_function(element(s,3))
        (x,y,z) -> f1(x)*f2(y)*f3(z)
    end
    # We should never get here
end
# Make a suitable function by undoing the map
function suitable_function(s::MappedDict)
    f = suitable_function(superdict(s))
    m = inv(mapping(s))
    x -> f(m*x)
end
function suitable_function(s::WeightedDict1d)
    f = suitable_function(superdict(s))
    g = weightfunction(s)
    x -> g(x) * f(x)
end
function suitable_function(s::WeightedDict2d)
    f = suitable_function(superdict(s))
    g = weightfunction(s)
    (x,y) -> g(x, y) * f(x, y)
end

suitable_interpolation_grid(basis::TensorProductDict) =
    ProductGrid(map(suitable_interpolation_grid, elements(basis))...)
suitable_interpolation_grid(basis::SineSeries) = MidpointEquispacedGrid(length(basis), 0, 1, domaintype(basis))
suitable_interpolation_grid(basis::WeightedDict) = suitable_interpolation_grid(superdict(basis))
suitable_interpolation_grid(basis::OperatedDict) = suitable_interpolation_grid(src_dictionary(basis))
