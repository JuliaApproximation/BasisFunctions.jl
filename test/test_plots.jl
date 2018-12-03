using BasisFunctions, BasisFunctions.Test, DomainSets

using Test, LinearAlgebra, Plots

delimit("Plots")
@testset begin
    e = random_expansion(FourierBasis(4))
    plot(e; plot_complex=false)
    plot(e; plot_complex=true)
    plot(e, exp)

    plot(random_expansion(FourierBasis(4)⊗FourierBasis(4)))

    plot(EquispacedGrid(4))

    plot(EquispacedGrid(4)×EquispacedGrid(4))

    plot(EquispacedGrid(4)×EquispacedGrid(4)×EquispacedGrid(4))

    plot(FourierBasis(4))

    plot(FourierBasis(4)[1])

    plot(FourierBasis(4,-3,5))

    plot(MultiDict((FourierBasis(4),FourierBasis(4))))
end
