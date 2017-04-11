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

      @test DiscreteGram(basis)*e ≈ evaluation_operator(basis)'evaluation_operator(basis)*e/n
    end
  end
end
# using BasisFunctions
# using Base.Test
#
# @testset begin discrete_gram_test(BigFloat) end
