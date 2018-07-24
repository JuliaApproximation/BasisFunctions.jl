
(VERSION < v"0.7-") ? nothing : Pkg.dev("https://github.com/MikaelSlevinsky/FastTransforms.jl") 
Pkg.clone("https://github.com/vincentcp/LinearAlgebra.jl.git")
Pkg.clone("https://github.com/daanhb/Domains.jl")
