# test_suite.jl
module test_suite

using Base.Test

using BasisFunctions
using FixedSizeArrays
using Debug
BF = BasisFunctions


########
# Auxiliary functions
########

# Keep track of successes, failures and errors
global failures = 0
global successes = 0
global errors = 0

const show_timings = true

# Custom test handler
custom_handler(r::Test.Success) = begin print_with_color(:green, "#\tSuccess "); println("on $(r.expr)"); global successes+=1;  end
custom_handler(r::Test.Failure) = begin print_with_color(:red, "\"\tFailure "); println("on $(r.expr)\""); global failures+=1; end
custom_handler(r::Test.Error) = begin println("\"\t$(typeof(r.err)) in $(r.expr)\""); global errors+=1; end
#custom_handler(r::Test.Error) = Base.showerror(STDOUT,r);

function message(y)
    println("\"\t",typeof(y),"\"")
    Base.showerror(STDOUT,y)
    global errors+=1
end

function message(y, backtrace)
    rethrow(y)
    println("\tFailure due to ",typeof(y))
    Base.showerror(STDOUT,y)
    Base.show_backtrace(STDOUT,backtrace)
    global errors+=1
end

function delimit(s::AbstractString)
    println("############")
    println("# ",s)
    println("############")
end


##########
# Testing
##########

include("test_generic_sets.jl")
include("test_generic_grids.jl")
include("test_generic_operators.jl")
include("test_ops.jl")
include("test_fourier.jl")
include("test_chebyshev.jl")



# Interpolate linearly between the left and right endpoint, using the value 0 <= scalar <= 1
function point_in_domain(basis::FunctionSet{1}, scalar)
    # Try to find an interval within the support of the basis
    a = left(basis)
    b = right(basis)

    T = numtype(basis)
    # Avoid infinities for some bases on the real line
    if isinf(a)
        a = -T(1)
    end
    if isinf(b)
        b = T(1)
    end
    x = scalar * a + (1-scalar) * b
end


# Interpolate linearly between the left and right endpoint, using the value 0 <= scalar <= 1
function point_in_domain(basis::FunctionSet, scalar)
    # Try to find an interval within the support of the basis
    a = left(basis)
    b = right(basis)

    # Avoid infinities for some bases on the real line
    va = Vector(a)
    vb = Vector(b)
    for i in 1:length(va)
        if isinf(va[i])
            va[i] = -1
        end
        if isinf(vb[i])
            vb[i] = 1
        end
    end
    pa = Vec(va...)
    pb = Vec(va...)
    x = scalar * pa + (1-scalar) * pb
end


function random_point_in_domain{N,T}(basis::FunctionSet{N,T})
    w = one(T) * rand()
    point_in_domain(basis, w)
end

function fixed_point_in_domain{N,T}(basis::FunctionSet{N,T})
    w = 1/sqrt(T(2))
    point_in_domain(basis, w)
end

function suitable_interpolation_grid(basis::FunctionSet)
    if BF.has_grid(basis)
        grid(basis)
    else
        EquispacedGrid(length(basis), point_in_domain(basis, 1), point_in_domain(basis, 0))
    end
end

suitable_interpolation_grid(basis::TensorProductSet) =
    TensorProductGrid(map(suitable_interpolation_grid, elements(basis))...)

suitable_interpolation_grid(basis::LaguerreBasis) = EquispacedGrid(length(basis), -10, 10)

suitable_interpolation_grid(basis::AugmentedSet) = suitable_interpolation_grid(set(basis))

random_index(basis::FunctionSet) = 1 + Int(floor(rand()*length(basis)))

Base.rationalize{N}(x::Vec{N,Float64}) = Vec{N,Rational{Int}}([rationalize(x_i) for x_i in x])

Base.rationalize{N}(x::Vec{N,BigFloat}) = Vec{N,Rational{BigInt}}([rationalize(x_i) for x_i in x])

function test_derived_sets(T)
    b1 = FourierBasis(11, T)
    b2 = ChebyshevBasis(12, T)

    delimit("Linear mapped sets")
    test_generic_set_interface(rescale(b1, -1, 2))

    delimit("Concatenated sets")
    test_generic_set_interface(b1 ⊕ b2)

    delimit("Operated sets")
    test_generic_set_interface(OperatedSet(differentiation_operator(b1)))

    delimit("Augmented sets")
    test_generic_set_interface(BF.Cos() * b1)
end




Test.with_handler(custom_handler) do


    for T in (Float64,BigFloat)

        println()
        println("T is ", T)

        SETS = (FourierBasis, ChebyshevBasis, ChebyshevBasisSecondKind, LegendreBasis,
                LaguerreBasis, HermiteBasis, PeriodicSplineBasis, CosineSeries)
#        SETS = (FourierBasis, ChebyshevBasis, ChebyshevBasisSecondKind, LegendreBasis,
#                LaguerreBasis, HermiteBasis, PeriodicSplineBasis, CosineSeries, SineSeries)
        for SET in SETS
            # Choose an odd and even number of degrees of freedom
            for n in (8, 11)
                basis = instantiate(SET, n, T)

                @test length(basis) == n
                @test numtype(basis) == T
                @test promote_type(eltype(basis),numtype(basis)) == eltype(basis)

                test_generic_set_interface(basis, SET)
            end
        end

        for basis in (FourierBasis(10) ⊗ ChebyshevBasis(12),
                        FourierBasis(11) ⊗ FourierBasis(20),
                        ChebyshevBasis(11) ⊗ ChebyshevBasis(20))
            test_generic_set_interface(basis, typeof(basis))
        end

        test_tensor_sets(T)

        test_tensor_operators(T)

        test_fourier_series(T)

        test_chebyshev(T)

        test_grids(T)

        test_ops(T)

        test_derived_sets(T)

    end # for T in...

end # Test.with_handler


# Diagnostics
println("Succes rate:\t$successes/$(successes+failures+errors)")
println("Failure rate:\t$failures/$(successes+failures+errors)")
println("Error rate:\t$errors/$(successes+failures+errors)")

end # module
