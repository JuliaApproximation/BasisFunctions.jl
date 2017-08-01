
function discrete_gram_test(T)
    for B in (ChebyshevBasis,FourierBasis,SineSeries,CosineSeries,BSplineTranslatesBasis,)
        basis = instantiate(B, 10, T)
        span = Span(basis)
        if !(B <: SineSeries)
            @test ! (typeof(DiscreteGram(span)) <: CompositeOperator)
        end
    end
    for n in (10,11), oversampling in 1:4
        e = map(T,rand(n))
        for B in (ChebyshevBasis,FourierBasis,SineSeries,CosineSeries,BSplineTranslatesBasis,)
            basis = instantiate(B, n, T)
            span = Span(basis)
            grid = BasisFunctions.oversampled_grid(basis, oversampling)
            @test DiscreteGram(span; oversampling=oversampling)*e ≈ evaluation_operator(span; oversampling=oversampling)'evaluation_operator(span, grid)*e/T(BasisFunctions.discrete_gram_scaling(basis, oversampling))
        end
    end
    for n in (10,11)
        e = map(T,rand(n))
        for B in (ChebyshevBasis,FourierBasis,BSplineTranslatesBasis)
            basis = instantiate(B, n, T)
            span = Span(basis)
            oversampling = 1
            @test n*(inv(evaluation_operator(span; oversampling=oversampling))')*e ≈ discrete_dual_evaluation_operator(span, oversampling=oversampling)*e
            for oversampling in 1:4
                grid = BasisFunctions.oversampled_grid(basis, oversampling)
                @test DiscreteDualGram(span; oversampling=oversampling)*e ≈ (discrete_dual_evaluation_operator(span; oversampling=oversampling)'discrete_dual_evaluation_operator(span; oversampling=oversampling))*e/T(BasisFunctions.discrete_gram_scaling(basis,oversampling))
                @test DiscreteMixedGram(span; oversampling=oversampling)*e ≈ (discrete_dual_evaluation_operator(span; oversampling=oversampling)'evaluation_operator(span; oversampling=oversampling))*e/T(BasisFunctions.discrete_gram_scaling(basis, oversampling))
            end
        end
    end
end

# using BasisFunctions
# using Base.Test
# #
# @testset begin discrete_gram_test(BigFloat) end
# #
