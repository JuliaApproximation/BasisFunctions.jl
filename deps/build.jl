if VERSION < v"0.7-"
    Pkg.clone("https://github.com/vincentcp/LinearAlgebra.jl.git")
    try
        Pkg.clone("https://github.com/JuliaApproximation/Domains.jl.git")
    catch y
        nothing
    finally
        Pkg.checkout("Domains", "julia-0.6-compatible")
    end
end
