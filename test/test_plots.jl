using BasisFunctions, BasisFunctions.Test, Domains

if VERSION < v"0.7-"
    using Base.Test, Plots
else
    using Test, LinearAlgebra, Plots
end

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
