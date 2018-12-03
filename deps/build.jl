
if VERSION < v"0.7-"
    Pkg.clone("https://github.com/vincentcp/LinearAlgebra.jl.git")
    try
        Pkg.clone("https://github.com/JuliaApproximation/DomainSets.jl.git")
    catch y
        nothing
    finally
        Pkg.checkout("DomainSets", "julia-0.6-compatible")
    end
else
    using Pkg
    if !in("GenericLinearAlgebra", keys(Pkg.installed()))
        Pkg.add(PackageSpec(url="http://github.com/andreasnoack/GenericLinearAlgebra.jl"))
    end
end
