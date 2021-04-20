
using BasisFunctions, DomainSets, Test
@testset "Spaces " begin
s = GenericFunctionSpace{Float64,ComplexF64}()
@test typeof(zero(s)(rand()))==ComplexF64
@test typeof(one(s)(rand()))==ComplexF64
@test zero(s)(rand())==0
@test one(s)(rand())==1

s = MeasureSpace(FourierWeight())
@test measure(s) ==FourierWeight()
@test domain(s) == UnitInterval()
@test space(FourierWeight()) == s
@test FourierSpace() == s
@test ChebyshevTSpace() == MeasureSpace(ChebyshevTWeight{Float64}())
@test Inf ∈ domain(L2())

s1 = GenericFunctionSpace{Float64,ComplexF64}()
s2 = MeasureSpace(FourierWeight())
@test components(tensorproduct(s1,s2)) == (s1,s2)
end
