# test_suite.jl
module test_suite

using Base.Test

using BasisFunctions
using FixedSizeArrays
BF = BasisFunctions


########
# Auxiliary functions
########

const show_timings = false


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

    @testset "$(rpad("Linear mapped sets",80))" begin
    test_generic_set_interface(rescale(b1, -1, 2)) end

    @testset "$(rpad("Operated sets",80))" begin
    test_generic_set_interface(OperatedSet(differentiation_operator(b1))) end

    @testset "$(rpad("Augmented sets",80))" begin
    test_generic_set_interface(BF.Cos() * b1) end

    @testset "$(rpad("Multiple sets",80))" begin
    test_generic_set_interface(multiset(b1,b2)) end

    @testset "$(rpad("A multiple and augmented set combination",80))" begin
    s = rescale(b1, 1/2, 1)
    test_generic_set_interface(multiset(s,Log()*s)) end
end




    for T in (Float64,BigFloat)
        println()
        delimit("T is $T", )
        delimit("Generic interfaces")

        SETS = (FourierBasis, ChebyshevBasis, ChebyshevBasisSecondKind, LegendreBasis,
                LaguerreBasis, HermiteBasis, PeriodicSplineBasis, CosineSeries)
        #        SETS = (FourierBasis, ChebyshevBasis, ChebyshevBasisSecondKind, LegendreBasis,
        #                LaguerreBasis, HermiteBasis, PeriodicSplineBasis, CosineSeries, SineSeries)
        @testset "$(rpad("$(name(instantiate(SET,n))) with $n dof",80," "))" for SET in SETS, n in (8,11)
            # Choose an odd and even number of degrees of freedom
                basis = instantiate(SET, n, T)

                @test length(basis) == n
                @test numtype(basis) == T
                @test promote_type(eltype(basis),numtype(basis)) == eltype(basis)

                test_generic_set_interface(basis, SET)
        end

        @testset "$(rpad("$(name(basis))",80," "))" for basis in (FourierBasis(10) ⊗ ChebyshevBasis(12),
                      FourierBasis(11) ⊗ FourierBasis(20),
                      ChebyshevBasis(11) ⊗ ChebyshevBasis(20))
            test_generic_set_interface(basis, typeof(basis))
        end

        delimit("Derived sets")
            test_derived_sets(T)

        delimit("Tensor specific tests")
        @testset "$(rpad("test iteration",80))" begin
            test_tensor_sets(T) end

        @testset "$(rpad("test operators",80))" begin
            test_tensor_operators(T) end

        delimit("Test Grids")
        @testset "$(rpad("Grids",80))" begin
            test_grids(T) end

        delimit("Check evaluations, interpolations, extensions, setexpansions")

        ## @testset "$(rpad("Fourier expansions",80))" begin
        ##     test_fourier_series(T) end

        @testset "$(rpad("Chebyshev expansions",80))" begin
            test_chebyshev(T) end

        @testset "$(rpad("Orthogonal polynomial evaluation",80))" begin
            test_ops(T) end

    end # for T in...
    println()
println(" All tests passed!")
end # module
