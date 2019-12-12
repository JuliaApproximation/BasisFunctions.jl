using PGFPlotsX, BasisFunctions

f = exp

F1 = Expansion( ChebyshevT(10), rand(ChebyshevT(10)))
F2 = Expansion( Fourier(10), rand(Fourier(10)))


for F in (F1,F2)
    Plot(F)

    Plot(F;n=300)

    Axis(F,f)
    Axis(F,F)
    Axis(f,F)
end

Plot(F2;plot_extension=true)
Axis(Plot(F2;plot_complex=true)...)
