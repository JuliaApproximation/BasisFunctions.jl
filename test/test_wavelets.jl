# test_wavelets.jl
exit()
using BasisFunctions
# Some semantic tests.
b1 = DaubechiesWaveletBasis(3,2)
b2 = CDFWaveletBasis(3,1,5)
b = CDFWaveletBasis(1,1,3)
eval_element(b, 1, 0.5)




using BasisFunctions
using Wavelets.DWT: wavelet, scaling
b1 = DaubechiesWaveletBasis(3,2)
b2 = CDFWaveletBasis(3,1,5)
b = CDFWaveletBasis(1,1,3)

eval_element(b, 1, 0.5)
println("Timings should be in the order of 0.001 seconds, 22.01k allocations, 625.094 KB")
@time begin
  for x in linspace(0,1,1000)
    eval_element(b, 1, x)
  end
end
supports = ((0,1),(0,1),(0.0,0.5),(0.5,1.0),(0.0,0.25),(0.25,0.5),(0.5,0.75),(0.75,1.0));
for i in 1:length(b)
  @assert left(b,i) == supports[i][1]
  @assert right(b,i) == supports[i][2]
end
for i in 1:length(b1)
  @assert support(b1,i) == (0.,1.)
end

@assert BasisFunctions.subset(b1,1:1) == BasisFunctions.DaubechiesWaveletBasis(3,1)
@assert BasisFunctions.subset(b2,1:3) == BasisFunctions.FunctionSubSet(b2,1:3)
@assert b2[1:5] == BasisFunctions.FunctionSubSet(b2,1:5)
@assert b2[1:4] == BasisFunctions.CDFWaveletBasis(3,1,4)
@assert BasisFunctions.dyadic_length(b1) == 2
@assert BasisFunctions.dyadic_length(b2) == 5
@assert length(b1) == 4
@assert length(b2) == 32
@assert promote_eltype(b1,Complex128) == DaubechiesWaveletBasis(3,2, Complex128)
@assert promote_eltype(b2,Complex128) == CDFWaveletBasis(3,1,5, Complex128)
@assert resize(b1,3) == BasisFunctions.DaubechiesWaveletBasis(3,3)
@assert BasisFunctions.name(b1) == "Basis of db3 wavelets"
@assert BasisFunctions.name(b2) == "Basis of cdf31 wavelets"
@assert native_index(b,1) == (scaling, 0, 0)
@assert native_index(b,2) == (wavelet, 0, 0)
@assert native_index(b,3) == (wavelet, 1, 0)
@assert native_index(b,4) == (wavelet, 1, 1)
@assert native_index(b,5) == (wavelet, 2, 0)
@assert native_index(b,6) == (wavelet, 2, 1)
@assert native_index(b,7) == (wavelet, 2, 2)
@assert native_index(b,8) == (wavelet, 2, 3)

for i in 1:length(b)
  @assert(linear_index(b,native_index(b,i))==i )
end
@assert grid(b1) == PeriodicEquispacedGrid(4,0,1)
@assert grid(b2) == PeriodicEquispacedGrid(32,0,1)
@assert BasisFunctions.period(b1)==1.

using Plots
b = CDFWaveletBasis(2,2,4)
gr()
plot(b)
