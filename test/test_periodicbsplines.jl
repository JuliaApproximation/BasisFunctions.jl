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
@assert matrix(BasisFunctions.Gram(b))*matrix(BasisFunctions.DualGram(b)) ≈ matrix(BasisFunctions.MixedGram(b))



b = FourierBasis(n)
I = eye(n)
@assert BasisFunctions.grammatrix(b) ≈ I
@assert BasisFunctions.dualgrammatrix(b) ≈ I
@assert matrix(BasisFunctions.Gram(b)) ≈ BasisFunctions.grammatrix(b)
@assert matrix(BasisFunctions.DualGram(b)) ≈ BasisFunctions.dualgrammatrix(b)
@assert matrix(BasisFunctions.Gram(b))*matrix(BasisFunctions.DualGram(b)) ≈ eye(n)
@assert matrix(BasisFunctions.Gram(b))*matrix(BasisFunctions.DualGram(b)) ≈ matrix(BasisFunctions.MixedGram(b))

using Plots
n = 20
b=PeriodicBSplineBasis(n,3)
t = linspace(0,1,1000)
f = map(x->BasisFunctions.eval_element(b,1,x),t)
  plot(t,f)


f = map(x->BasisFunctions.eval_dualelement(b,1,x),t)
# f = map(x->BasisFunctions.eval_expansion(b,[1,0,0,0,0],x),t)
  plot!(t,f,ylims=[-n,2n],linestyle=:dot)

@which eval_expansion(b,ones(n),t)


f = map(x->BasisFunctions.eval_expansion(b,ones(n),x),t)
# f = map(x->BasisFunctions.eval_expansion(b,[1,0,0,0,0,0,0,0,0,0],x),t)
  plot!(t,f,ylims=[-n,2n])


exit()
using BasisFunctions
using FrameFun
opts = [(:reltol,1e-9), (:abstol,1e-9)]
n = 3
b = FourierBasis(n,-1,1)
G = Gram(b; reltol=1e-9, abstol=1e-9)
GD = DualGram(b; reltol=1e-9, abstol=1e-9)
GM = MixedGram(b; reltol=1e-9, abstol=1e-9)

frame = FrameFun.extensionframe(Interval(left(b)+eps(Float64),right(b)-eps(Float64))*.5,b)
tol = .25
# frame = FrameFun.extensionframe(Interval(left(b)+tol,right(b)-tol),b)
G = Gram(frame; reltol=1e-9, abstol=1e-9)
GD = DualGram(frame; reltol=1e-9, abstol=1e-9)
GM = MixedGram(frame; reltol=1e-9, abstol=1e-9)
matrix(G*GD)
@assert norm(eye(n)-real(matrix(G)))<1e-10
@assert norm(imag(matrix(G)))<1e-10
@assert norm(eye(n)-real(matrix(GD)))<1e-10
@assert norm(imag(matrix(GD)))<1e-10
@assert norm(eye(n)-real(matrix(GM)))<1e-6
@assert norm(imag(matrix(GM)))<1e-10

BasisFunctions.project(b, x->b[2](x); reltol=1e-9, abstol=1e-9)
BasisFunctions.project(frame, x->b[2](x); reltol=1e-9, abstol=1e-9)

DAO = BasisFunctions.approximation_operator(b; discrete=true, opts...)
CAO = BasisFunctions.approximation_operator(b; discrete=false, opts...)
FDAO = BasisFunctions.approximation_operator(frame; discrete=true, opts...)
FCAO = BasisFunctions.approximation_operator(frame; discrete=false, opts...)

f = x->b[2](x)
bb = BasisFunctions.project(b,f; opts...)
Fb = BasisFunctions.project(frame, f; opts...)
CAO*bb
CAO*f

FC = approximate(b, f; discrete=false, opts...)
FD = approximate(b, f; discrete=true, opts...)
FCF = approximate(frame, f; discrete=false, opts...)
FDF = approximate(frame, f; discrete=true, opts...)

coefficients(FCF)
  println(coefficients(FCF))
coefficients(FDF)

using Plots

gr()
plot(FC; layout = 4, ylims=[-1,3])
plot!(FD; subplot = 2, ylims=[-1,3])
plot!(FCF; subplot = 3, ylims=[-1,3])
plot!(FDF; subplot = 4, ylims=[-1,3])
