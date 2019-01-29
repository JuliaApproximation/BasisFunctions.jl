function pinv(C::Circulant{T}, tolerance::T = eps(T)) where T<:Real
    vdft = copy(C.vcvr_dft)
    vdft[abs.(vdft).<tolerance] .= Inf
    vdft .= 1 ./ vdft
    return Circulant(real(C.dft \ vdft), copy(vdft), similar(vdft), C.dft)
end

function pinv(C::Circulant, tolerance::Real = eps(real(T)))
    vdft = copy(C.vcvr_dft)
    vdft[abs.(vdft).<tolerance] .= Inf
    vdft .= 1 ./ vdft
    return Circulant(C.dft \ vdft, copy(vdft), similar(vdft), C.dft)
end

eigvals(C::Circulant) = copy(C.vcvr_dft)
sqrt(C::Circulant{T}) where T<:Real = Circulant(real(ifft(sqrt.(C.vcvr_dft))))
sqrt(C::Circulant) = Circulant(ifft(sqrt.(C.vcvr_dft)))
Base.copy(C::Circulant) = Circulant(copy(C.vc))
Base.similar(C::Circulant) = Circulant(similar(C.vc))

function (+)(C1::Circulant, C2::Circulant)
    @boundscheck (size(C1)==size(C2)) || throw(BoundsError())
    Circulant(C1.vc+C2.vc)
end

function (-)(C1::Circulant, C2::Circulant)
    @boundscheck (size(C1)==size(C2)) || throw(BoundsError())
    Circulant(C1.vc-C2.vc)
end

(-)(C::Circulant) = Circulant(-C.vc)

function (*)(C1::Circulant, C2::Circulant)
    @boundscheck (size(C1)==size(C2)) || throw(BoundsError())
    Circulant(ifft(C1.vcvr_dft.*C2.vcvr_dft))
end

(*)(scalar::Number, C::Circulant) = Circulant(scalar*C.vc)
