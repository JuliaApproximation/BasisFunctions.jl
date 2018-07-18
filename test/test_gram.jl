using BasisFunctions
if VERSION < v"0.7-"
    using Base.Test
else
    using Test
end

function discrete_gram_test(T)
    for B in (ChebyshevBasis,FourierBasis,SineSeries,CosineSeries)#,BSplineTranslatesBasis,)
        basis = instantiate(B, 10, T)
        span = basis
        if !(B <: SineSeries)
            @test ! (typeof(DiscreteGram(span)) <: CompositeOperator)
        end
    end
    # Had to add these lines to get the terminal to run without errors. No idea why. VC
    n = 10
    B = ChebyshevBasis
    oversampling = 1
    basis = instantiate(B, n, T)
    ##################################################################################
    for n in (10,11), oversampling in 1:4
        e = map(T,rand(n))
        for B in (ChebyshevBasis,FourierBasis,SineSeries,CosineSeries)#,BSplineTranslatesBasis,)
            basis = instantiate(B, n, T)
            span = basis
            grid = BasisFunctions.oversampled_grid(basis, oversampling)
            @test DiscreteGram(span; oversampling=oversampling)*e ≈ evaluation_operator(span; oversampling=oversampling)'evaluation_operator(span, grid)*e/T(BasisFunctions.discrete_gram_scaling(basis, oversampling))
        end
    end
    for n in (10,11)
        e = map(T,rand(n))
        for B in (ChebyshevBasis,FourierBasis)#,BSplineTranslatesBasis)
            basis = instantiate(B, n, T)
            span = basis
            oversampling = 1
            @test n*(inv(evaluation_operator(span; oversampling=oversampling, sparse=false))')*e ≈ discrete_dual_evaluation_operator(span, oversampling=oversampling)*e
            for oversampling in 1:4
                grid = BasisFunctions.oversampled_grid(basis, oversampling)
                @test DiscreteDualGram(span; oversampling=oversampling)*e ≈ (discrete_dual_evaluation_operator(span; oversampling=oversampling)'discrete_dual_evaluation_operator(span; oversampling=oversampling))*e/T(BasisFunctions.discrete_gram_scaling(basis,oversampling))
                @test DiscreteMixedGram(span; oversampling=oversampling)*e ≈ (discrete_dual_evaluation_operator(span; oversampling=oversampling)'evaluation_operator(span; oversampling=oversampling))*e/T(BasisFunctions.discrete_gram_scaling(basis, oversampling))
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


#
# @testset begin discrete_gram_test(Float64) end
# @testset begin general_gram_test(Float64) end
# #
