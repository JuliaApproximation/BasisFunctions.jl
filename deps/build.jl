if VERSION < v"0.7-"
    Pkg.clone("https://github.com/vincentcp/LinearAlgebra.jl.git")
    try
        Pkg.clone("https://github.com/JuliaApproximation/DomainSets.jl.git")
    catch y
        nothing
    finally
        Pkg.checkout("DomainSets", "julia-0.6-compatible")
    end
end
