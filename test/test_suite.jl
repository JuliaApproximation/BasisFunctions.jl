# test_suite.jl
module test_suite

using Base.Test

using BasisFunctions
using StaticArrays
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
include("test_maps.jl")



# Interpolate linearly between the left and right endpoint, using the value 0 <= scalar <= 1
function point_in_domain(basis::FunctionSet1d, scalar)
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
    x = (1-scalar) * a + scalar * b
end


# Interpolate linearly between the left and right endpoint, using the value 0 <= scalar <= 1
function point_in_domain(basis::FunctionSet, scalar)
    # Try to find an interval within the support of the basis
    a = left(basis)
    b = right(basis)

    if isinf(norm(a)) || isinf(norm(b))
        va = MVector(a)
        vb = MVector(b)
        # Avoid infinities for some bases on the real line
        for i in 1:length(va)
            if isinf(va[i])
                va[i] = -1
            end
            if isinf(vb[i])
                vb[i] = 1
            end
        end
        a = SVector(va)
        b = SVector(vb)
    end
    x = (1-scalar) * a + scalar * b
end

# Abuse point_in_domain with a scalar greater than one in order to get
# a point outside the domain.
point_outside_domain(basis::FunctionSet) = point_in_domain(basis, eltype(basis)(1.1))

point_outside_domain(basis::LaguerreBasis) = -one(eltype(basis))
point_outside_domain(basis::HermiteBasis) = one(eltype(basis))+im

function random_point_in_domain(basis::FunctionSet)
    T = eltype(basis)
    w = one(T) * rand()
    point_in_domain(basis, w)
end

function fixed_point_in_domain(basis::FunctionSet)
    T = eltype(basis)
    w = 1/sqrt(T(2))
    point_in_domain(basis, w)
end

function suitable_interpolation_grid(basis::FunctionSet)
    if BF.has_grid(basis)
        grid(basis)
    else
        T = numtype(basis)
        EquispacedGrid(length(basis), point_in_domain(basis, T(0)), point_in_domain(basis, T(1)))
    end
end

suitable_interpolation_grid(basis::TensorProductSet) =
    TensorProductGrid(map(suitable_interpolation_grid, elements(basis))...)

suitable_interpolation_grid(basis::LaguerreBasis) = EquispacedGrid(length(basis), 0, 10, numtype(basis))

suitable_interpolation_grid(basis::SineSeries) = MidpointEquispacedGrid(length(basis), -1, 1, numtype(basis))

suitable_interpolation_grid(basis::AugmentedSet) = suitable_interpolation_grid(set(basis))

random_index(basis::FunctionSet) = 1 + Int(floor(rand()*length(basis)))

Base.rationalize{N}(x::SVector{N,Float64}) = SVector{N,Rational{Int}}([rationalize(x_i) for x_i in x])

Base.rationalize{N}(x::SVector{N,BigFloat}) = SVector{N,Rational{BigInt}}([rationalize(x_i) for x_i in x])

function test_derived_sets(T)
    b1 = FourierBasis(11, T)
    b2 = ChebyshevBasis(12, T)

    @testset "$(rpad("Generic derived set",80))" begin
    test_generic_set_interface(BasisFunctions.ConcreteDerivedSet(b2)) end

    @testset "$(rpad("Linear mapped sets",80))" begin
    test_generic_set_interface(rescale(b1, -T(1), T(2)))
    end

    @testset "$(rpad("A simple subset",80))" begin
    test_generic_set_interface(b1[2:6]) end

    @testset "$(rpad("Operated sets",80))" begin
    test_generic_set_interface(OperatedSet(differentiation_operator(b1))) end

    @testset "$(rpad("Augmented sets",80))" begin
    # Try a functor
    test_generic_set_interface(BF.Cos() * b1)
    # as well as a regular function
    test_generic_set_interface(cos * b1) end

    @testset "$(rpad("Multiple sets",80))" begin
    test_generic_set_interface(multiset(b1,b2))
    test_generic_set_interface(MultiSet((b1,b2))) end

    @testset "$(rpad("A multiple and augmented set combination",80))" begin
    s = rescale(b1, 1/2, 1)
    test_generic_set_interface(multiset(s,Log()*s)) end

    @testset "$(rpad("A complicated subset",80))" begin
    s = rescale(b1, 1/2, 1)
    test_generic_set_interface(s[1:5]) end

    @testset "$(rpad("Piecewise sets",80))" begin
    part = PiecewiseInterval(0, 10, 10)
    pw = PiecewiseSet(b2, part)
    test_generic_set_interface(pw) end

    # @testset "$(rpad("A tensor product of MultiSet's",80))" begin
    # b = multiset(b1,b2)
    # c = b ⊗ b
    # test_generic_set_interface(c) end
end


    # Verify types of FFT and DCT plans by FFTW
    # If anything changes here, the aliases in fouriertransforms.jl have to change as well
    d1 = plan_fft!(zeros(Complex{Float64}, 10), 1:1)
    @test typeof(d1) == Base.DFT.FFTW.cFFTWPlan{Complex{Float64},-1,true,1}
    d2 = plan_fft!(zeros(Complex{Float64}, 10, 10), 1:2)
    @test typeof(d2) == Base.DFT.FFTW.cFFTWPlan{Complex{Float64},-1,true,2}
    d3 = plan_bfft!(zeros(Complex{Float64}, 10), 1:1)
    @test typeof(d3) == Base.DFT.FFTW.cFFTWPlan{Complex{Float64},1,true,1}
    d4 = plan_bfft!(zeros(Complex{Float64}, 10, 10), 1:2)
    @test typeof(d4) == Base.DFT.FFTW.cFFTWPlan{Complex{Float64},1,true,2}

    d5 = plan_dct!(zeros(10), 1:1)
    @test typeof(d5) == Base.DFT.FFTW.DCTPlan{Float64,5,true}
    d6 = plan_idct!(zeros(10), 1:1)
    @test typeof(d6) == Base.DFT.FFTW.DCTPlan{Float64,4,true}

    for T in (Float64,BigFloat)
        println()
        delimit("T is $T", )
        delimit("Operators")
        @testset "$(rpad("test diagonal operators",80))" begin
            test_diagonal_operators(T) end

        @testset "$(rpad("test multidiagonal operators",80))" begin
            test_multidiagonal_operators(T) end

        @testset "$(rpad("test invertible operators",80))" begin
            test_invertible_operators(T) end

        @testset "$(rpad("test noninvertible operators",80))" begin
            test_noninvertible_operators(T) end

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

        delimit("Test Maps")
        @testset "$(rpad("Maps",80))" begin
            test_maps(T) end

        delimit("Check evaluations, interpolations, extensions, setexpansions")

        @testset "$(rpad("Fourier expansions",80))" begin
            test_fourier_series(T) end

        @testset "$(rpad("Chebyshev expansions",80))" begin
            test_chebyshev(T) end

        @testset "$(rpad("Orthogonal polynomial evaluation",80))" begin
            test_ops(T) end

    end # for T in...
    println()
println(" All tests passed!")
end # module
