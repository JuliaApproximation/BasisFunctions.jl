using BasisFunctions, BasisFunctions.Test

using Test

types = (Float64, BigFloat)

function gram_projection_test(T)
    for B in (ChebyshevBasis, FourierBasis, SineSeries, CosineSeries, LegendrePolynomials)
        basis = instantiate(B, 10, T)
        if !(B <: SineSeries)
            @test ! (typeof(DiscreteGram(basis)) <: CompositeOperator)
        end
    end
    # Had to add these lines to get the terminal to run without errors. No idea why. VC
    n = 10
    B = ChebyshevBasis
    oversampling = 1
    basis = instantiate(B, n, T)
    ##################################################################################
    for n in (10,11), oversampling in 1:4
        e = rand(T, n)
        for B in (ChebyshevBasis,FourierBasis,SineSeries,CosineSeries)#,BSplineTranslatesBasis,)
            basis = instantiate(B, n, T)
            grid = BasisFunctions.oversampled_grid(basis, oversampling)
            @test DiscreteGram(basis; oversampling=oversampling)*e ≈ evaluation_operator(basis; oversampling=oversampling)'evaluation_operator(basis, grid)*e/T(BasisFunctions.discrete_gram_scaling(basis, oversampling))
        end
    end
    for n in (10,11)
        e = rand(T, n)
        for B in (ChebyshevBasis,FourierBasis)#,BSplineTranslatesBasis)
            basis = instantiate(B, n, T)
            oversampling = 1
            @test n*(inv(evaluation_operator(basis; oversampling=oversampling, sparse=false))')*e ≈ discrete_dual_evaluation_operator(basis, oversampling=oversampling)*e
            for oversampling in 1:4
                grid = BasisFunctions.oversampled_grid(basis, oversampling)
                @test DiscreteDualGram(basis; oversampling=oversampling)*e ≈ (discrete_dual_evaluation_operator(basis; oversampling=oversampling)'discrete_dual_evaluation_operator(basis; oversampling=oversampling))*e/T(BasisFunctions.discrete_gram_scaling(basis,oversampling))
                @test DiscreteMixedGram(basis; oversampling=oversampling)*e ≈ (discrete_dual_evaluation_operator(basis; oversampling=oversampling)'evaluation_operator(basis; oversampling=oversampling))*e/T(BasisFunctions.discrete_gram_scaling(basis, oversampling))
            end
        end
    end
end

function general_gram_test(T)
    tol = max(sqrt(eps(T)), 1e-10)
    for method in (Gram, DualGram, MixedGram), B in (FourierBasis{T}(11), )#BSplineTranslatesBasis(5, 1,T))
        GBB = method(B,B; atol=tol, rtol=tol)
        GB = method(B; atol=tol, rtol=tol)

        e = rand(length(B))
        @test norm(GBB*e - GB*e) <= 1000*tol
    end
end

for T in types
    @testset "$(rpad("Gram functionality $T",80))" begin
        gram_projection_test(T)
    end
end
