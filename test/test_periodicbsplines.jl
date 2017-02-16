exit()
using BasisFunctions

b = PeriodicBSplineBasis(5,3)

@assert degree(b) == 3
@assert length(b)==5
@assert BasisFunctions.name(b) == "Periodic squeezed B-splines of degree 3"
@assert left(b) == 0
@assert right(b)==1
@assert left(b,1)≈ 0
@assert left(b,2)≈1/5
@assert left(b,3)≈2/5
@assert left(b,4)≈3/5
@assert left(b,5)≈4/5
@assert right(b,1)≈4/5
@assert right(b,2)≈5/5
@assert right(b,3)≈6/5
@assert right(b,4)≈7/5
@assert right(b,5)≈8/5
@assert instantiate(PeriodicBSplineBasis, 4, Float16)==PeriodicBSplineBasis(4,3,Float16)
@assert set_promote_eltype(b, Float16)==PeriodicBSplineBasis(5,3,Float16)
@assert set_promote_eltype(b, complex(Float64))==PeriodicBSplineBasis(5,3,Complex128)
@assert resize(b, 20)==PeriodicBSplineBasis(20,3)
@assert grid(b)==PeriodicEquispacedGrid(5,0,1)
@assert BasisFunctions.period(b)==1
@assert BasisFunctions.stepsize(b)==1/5
@assert in_support(b,1,.5)
@assert !in_support(b,1,.81)
@assert !in_support(b,1,.99)
@assert in_support(b,3,.2)
@assert in_support(b,3,.4)
@assert !in_support(b,3,.21)
@assert !in_support(b,3,.39)

n = 3
b=PeriodicBSplineBasis(n,1)
@assert BasisFunctions.grammatrix(b) ≈ [2/3 1/6 1/6; 1/6 2/3 1/6;1/6 1/6 2/3]*BasisFunctions.splinescaling(n)^2/n
@assert BasisFunctions.dualgrammatrix(b) ≈ [5/3 -1/3 -1/3; -1/3 5/3 -1/3; -1/3 -1/3 5/3]*n/BasisFunctions.splinescaling(n)^2
@assert matrix(BasisFunctions.Gram(b)) ≈ BasisFunctions.grammatrix(b)
@assert matrix(BasisFunctions.DualGram(b)) ≈ BasisFunctions.dualgrammatrix(b)
@assert matrix(BasisFunctions.Gram(b))*matrix(BasisFunctions.DualGram(b)) ≈ eye(n)


BasisFunctions.eval_element(b,1,0)^2
matrix(BasisFunctions.Gram(b))






matrix(BasisFunctions.DualGram(b))




using Plots
n = 10
b=PeriodicBSplineBasis(n,1)
t = linspace(0,1,1000)
f = map(x->BasisFunctions.eval_element(b,1,x),t)
  plot(t,f)


f = map(x->BasisFunctions.eval_dualelement(b,1,x),t)
# f = map(x->BasisFunctions.eval_expansion(b,[1,0,0,0,0],x),t)
  plot!(t,f,ylims=[-n,2n],linestyle=:dot)


f = map(x->BasisFunctions.eval_expansion(b,ones(n),x),t)
# f = map(x->BasisFunctions.eval_expansion(b,[1,0,0,0,0],x),t)
  plot(t,f,ylims=[0,2n])
