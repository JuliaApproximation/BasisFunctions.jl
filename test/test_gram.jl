function discrete_gram_test(T)
  for B in (ChebyshevBasis,FourierBasis,SineSeries,CosineSeries,BSplineTranslatesBasis,)
    basis = instantiate(B, 10, T)
    if !(B <: SineSeries)
      @test ! (typeof(DiscreteGram(basis))<:CompositeOperator)
    end
  end
  for n in (10,11), oversampling in 1:4
    e = map(T,rand(n))
    for B in (ChebyshevBasis,FourierBasis,SineSeries,CosineSeries,BSplineTranslatesBasis,)
      basis = instantiate(B, n, T)
      grid = BasisFunctions.oversampled_grid(basis,oversampling)
      @test DiscreteGram(basis; oversampling=oversampling)*e ≈ evaluation_operator(basis; oversampling=oversampling)'evaluation_operator(basis, grid)*e/length(grid)
    end
  end
  for n in (10,11)
    e = map(T,rand(n))
    for B in (ChebyshevBasis,FourierBasis,BSplineTranslatesBasis)
      basis = instantiate(B, n, T)
      oversampling = 1
      @test n*(inv(evaluation_operator(basis; oversampling=oversampling))')*e ≈ discrete_dual_evaluation_operator(basis, oversampling=oversampling)*e
      for oversampling in 1:4
        grid = BasisFunctions.oversampled_grid(basis, oversampling)
        @test DiscreteDualGram(basis; oversampling=oversampling)*e ≈ (discrete_dual_evaluation_operator(basis; oversampling=oversampling)'discrete_dual_evaluation_operator(basis; oversampling=oversampling))*e/length(grid)
        @test DiscreteMixedGram(basis; oversampling=oversampling)*e ≈ (discrete_dual_evaluation_operator(basis; oversampling=oversampling)'evaluation_operator(basis; oversampling=oversampling))*e/length(grid)
      end
    end
  end

end

# using BasisFunctions
# using Base.Test
# #
# @testset begin discrete_gram_test(Float64) end
# #
