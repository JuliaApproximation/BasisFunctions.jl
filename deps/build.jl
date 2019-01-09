
using Pkg
if !in("GenericLinearAlgebra", keys(Pkg.installed()))
    Pkg.add(PackageSpec(url="http://github.com/andreasnoack/GenericLinearAlgebra.jl"))
end
