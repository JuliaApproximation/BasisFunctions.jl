# waveletbasis.jl

abstract WaveletBasis{T} <: AbstractBasis1d{T}


hascompactsupport{B <: WaveletBasis}(::Type{B}) = True


abstract OrthogonalWaveletBasis{T} <: WaveletBasis{T}

abstract BiorthogonalWaveletBasis{T} <: WaveletBasis{T}


