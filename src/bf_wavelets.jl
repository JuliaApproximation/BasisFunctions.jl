# bf_wavelets.jl

abstract WaveletBasis{T} <: FunctionSet1d{T}


abstract OrthogonalWaveletBasis{T} <: WaveletBasis{T}

abstract BiorthogonalWaveletBasis{T} <: WaveletBasis{T}
