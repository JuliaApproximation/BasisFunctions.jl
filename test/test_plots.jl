using BasisFunctions, BasisFunctions.Test, DomainSets, GridArrays

using Test, LinearAlgebra, PGFPlotsX, Plots

delimit("Plots")
@testset begin
    e = random_expansion(Fourier(4))
    plot(e; plot_complex=false)
    plot(e; plot_complex=true)
    plot(e, exp)

    plot(random_expansion(Fourier(4)⊗Fourier(4)))

    plot(Fourier(4))

    plot(Fourier(4)[1])

    plot(Fourier(4) → -3..5)

    plot(MultiDict((Fourier(4),Fourier(4))))
end
