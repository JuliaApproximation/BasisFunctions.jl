# test_wavelets.jl
using BasisFunctions
    BF = BasisFunctions
    using Base.Test
    using WaveletsCopy.DWT: wavelet, scaling, db3,  db4, Primal
    using StaticArrays
    using BasisFunctions: wavelet_dual
    using Domains
    try
        test_generic_dict_interface
    catch
        include("test_generic_dicts.jl")
        include("util_functions.jl")
    end
    suitable_function(set::BasisFunctions.WaveletBasis) =  x -> 1/(10+cos(2*pi*x))

    BasisFunctions.has_transform(::WaveletBasis) = false

    supports_interpolation(::WaveletBasis) = false


using WaveletsCopy.DWT: quad_trap, quad_sf, quad_sf_weights, quad_sf_N, quad_trap_N
function test_wavelet_quadrature()
        @testset begin
            M1 = 5; M2 = 10; J = 3
            wav = db3; g = x->sin(2pi*x)
            reference = quad_sf_N(g, wav, M2, J, 5)

            b = DaubechiesWaveletBasis(3,J)
            S = BasisFunctions.DWTSamplingOperator(b, Int(M1/5),0)
            @test norm(S*g-reference/sqrt(1<<J)) < 1e-3
            S = BasisFunctions.DWTSamplingOperator(b, Int(M2/5), 0)
            @test norm(S*g-reference/sqrt(1<<J)) < 1e-8
            S = BasisFunctions.DWTSamplingOperator(b, Int(M1/5),7)
            @test norm(S*g-reference/sqrt(1<<J)) < 1e-15
            S = BasisFunctions.DWTSamplingOperator(b, Int(M2/5), 3)
            @test norm(S*g-reference/sqrt(1<<J)) < 1e-15
        end
end

test_wavelet_quadrature()


function bf_wavelets_implementation_test()
    @testset begin
        test_generic_dict_interface(CDFWaveletBasis(3,5,6))
        test_generic_dict_interface(wavelet_dual(CDFWaveletBasis(3,5,6)))
        test_generic_dict_interface(CDFScalingBasis(1,1,6))

        b1 = DaubechiesWaveletBasis(3,2)
        b2 = CDFWaveletBasis(3,1,5)
        b = CDFWaveletBasis(1,1,3)
        BasisFunctions.unsafe_eval_element(b, 1, .1)
        624 == @allocated BasisFunctions.unsafe_eval_element(b, 1, .1)
        supports = ((0.,1.),(0.,1.),(0.0,0.5),(0.5,1.0),(0.0,0.25),(0.25,0.5),(0.5,0.75),(0.75,1.0));
        for i in ordering(b)
            @test infimum(support(b,i)) == supports[value(i)][1]
            @test supremum(support(b,i)) == supports[value(i)][2]
        end
        for i in ordering(b1)
            @test support(b1,i) == UnitInterval()
        end
        @test BasisFunctions.subdict(b1,1:1) == BasisFunctions.DaubechiesWaveletBasis(3,0)
        @test BasisFunctions.subdict(b2,1:3) == BasisFunctions.LargeSubdict(b2,1:3)
        @test b2[1:5] == BasisFunctions.LargeSubdict(b2,1:5)
        @test b2[1:4] == BasisFunctions.CDFWaveletBasis(3,1,2)
        @test BasisFunctions.dyadic_length(b1) == 2
        @test BasisFunctions.dyadic_length(b2) == 5
        @test length(b1) == 4
        @test length(b2) == 32
        @test BasisFunctions.dict_promote_domaintype(b1,Complex128) == DaubechiesWaveletBasis(3,2, Complex128)
        @test BasisFunctions.dict_promote_domaintype(b2, Complex128) == CDFWaveletBasis(3,1,5, BasisFunctions.Prl, Complex128)
        @test resize(b1,8) == BasisFunctions.DaubechiesWaveletBasis(3,3)
        @test BasisFunctions.name(b1) == "Basis of db3 wavelets"
        @test BasisFunctions.name(b2) == "Basis of cdf31 wavelets"
        @test BasisFunctions.native_index(b,1) == (scaling, 0, 0)
        @test BasisFunctions.native_index(b,2) == (wavelet, 0, 0)
        @test BasisFunctions.native_index(b,3) == (wavelet, 1, 0)
        @test BasisFunctions.native_index(b,4) == (wavelet, 1, 1)
        @test BasisFunctions.native_index(b,5) == (wavelet, 2, 0)
        @test BasisFunctions.native_index(b,6) == (wavelet, 2, 1)
        @test BasisFunctions.native_index(b,7) == (wavelet, 2, 2)
        @test BasisFunctions.native_index(b,8) == (wavelet, 2, 3)
        for i in 1:length(b)
            @test(linear_index(b,BasisFunctions.native_index(b,i))==i )
        end
        @test grid(b1) == PeriodicEquispacedGrid(4,0,1)
        @test grid(b2) == PeriodicEquispacedGrid(32,0,1)
        @test BasisFunctions.period(b1)==1.

        # test grid eval functions
        for g in (plotgrid(b,200), PeriodicEquispacedGrid(128,0,1))
            for i in ordering(b)
                tic(); e1 = BasisFunctions._default_unsafe_eval_element_in_grid(b, i, g); t1 = toq();
                tic(); e2 = BasisFunctions._unsafe_eval_element_in_dyadic_grid(b, i, g); t2 = toq();
            end
            for i in ordering(b)
                tic(); e1 = BasisFunctions._default_unsafe_eval_element_in_grid(b, i, g); t1 = toq();
                tic(); e2 = BasisFunctions._unsafe_eval_element_in_dyadic_grid(b, i, g); t2 = toq();
                @test e1 ≈ e2
                @test t2 < t1
            end
        end

        b1 = DaubechiesWaveletBasis(3,6)
        b2 = CDFWaveletBasis(3,1,5)
        b = CDFWaveletBasis(1,1,6)
        for basis in (b,b1,b2,BasisFunctions.wavelet_dual(b), ScalingBasis(b))
            @test evaluation_matrix(basis, BasisFunctions.grid(basis))≈evaluation_matrix(basis, collect(BasisFunctions.grid(basis)))
            T = Float64
            ELT = Float64
            tbasis = transform_dict(basis)
            x = ones(length(basis))
            t = transform_operator(tbasis, basis)
            it = transform_operator(basis, tbasis)
            @test norm((it*t*x-x))+1≈1
        end
    end
end
bf_wavelets_implementation_test()

using Plots
b = CDFWaveletBasis(1,5,3)
plot(b,layout=2)
plot(wavelet_dual(b)[3:4],layout=2,subplot=1)
plot!(BasisFunctions.Dual, wavelet, wavelet(b),j=1,k=0,subplot=2,periodic=true)
plot!(BasisFunctions.Dual, wavelet, wavelet(b),j=1,k=1,subplot=2,periodic=true)
BasisFunctions.has_transform(::WaveletBasis) = true
