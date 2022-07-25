
using BasisFunctions: iswidertype

@test iswidertype(Float32,Float32)
@test iswidertype(Float32,Float64)
@test iswidertype(Float32,BigFloat)
@test iswidertype(Float64,Float64)
@test iswidertype(Float64,BigFloat)

@test iswidertype(Float64,Complex{Float64})
@test iswidertype(Float64,Complex{BigFloat})
