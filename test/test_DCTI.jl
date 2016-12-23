
using BasisFunctions
using Base.Test

n = 121
# Check some implementation
full_transform_operator(DiscreteGridSpace(ChebyshevNodeGrid(n)), ChebyshevBasis(n))
full_transform_operator(ChebyshevBasis(n), nodegrid=true)
full_transform_operator(ChebyshevBasis(n), nodegrid=false)
function test_full_transform_extremagrid()
  for n in (10,101)
    for T in (Float32, Float64, Complex64, Complex128)
      coef = zeros(T,n)
      # tol = 10^(4/5*log10(eps(real(T))))
      tol = sqrt(eps(real(T)))
      for i in 1:n
        B = ChebyshevBasis(n,T)
        G = BasisFunctions.secondgrid(B)
        coef[i] = 1
        SE = SetExpansion(B,coef)
        O = full_transform_operator(B, nodegrid=false)

        f1 = coefficients(O*SE)
        f2 = BasisFunctions.broadcast(SE,G)
        @test(norm(f2-f1) < tol)
        coef[i] = 0
      end
    end
  end
end


function test_inverse_transform_extremagrid()
  for n in (10,101)
    for T in (Float32, Float64, Complex64, Complex128)
      coef = zeros(T,n)
      # tol = 10^(4/5*log10(eps(real(T))))
      tol = sqrt(eps(real(T)))
      for i in 1:n
        B = ChebyshevBasis(n,T)
        G = BasisFunctions.secondgrid(B)
        coef[i] = 1
        SE = SetExpansion(B,coef)
        O = full_transform_operator(B, nodegrid=false)
        Oinv = full_transform_operator(dest(O), src(O))
        c = coefficients(Oinv*O*SE)
        @test(norm(coef-c) < tol)
        coef[i] = 0
      end
    end
  end
end
