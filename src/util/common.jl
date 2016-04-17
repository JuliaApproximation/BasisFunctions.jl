# common.jl

# Convenience definitions for the implementation of traits
typealias True Val{true}
typealias False Val{false}

# Arithmetic with such traits
(&){T1,T2}(::Type{Val{T1}}, ::Type{Val{T2}}) = Val{T1 & T2}
(|){T1,T2}(::Type{Val{T1}}, ::Type{Val{T2}}) = Val{T1 | T2}

(&){T1,T2}(::Val{T1}, ::Val{T2}) = Val{T1 & T2}()
(|){T1,T2}(::Val{T1}, ::Val{T2}) = Val{T1 | T2}()

"Return a complex type associated with the argument type."
complexify{T <: Real}(::Type{T}) = Complex{T}
complexify{T <: Real}(::Type{Complex{T}}) = Complex{T}
# In 0.5 we will be able to use Base.complex(T)
isreal{T <: Real}(::Type{T}) = True
isreal{T <: Real}(::Type{Complex{T}}) = False

# Starting with julia 0.4.3 we can just do float(T)
floatify{T <: AbstractFloat}(::Type{T}) = T
floatify(::Type{Int}) = Float64
floatify(::Type{BigInt}) = BigFloat
floatify{T}(::Type{Complex{T}}) = Complex{floatify(T)}
floatify{T}(::Type{Rational{T}}) = floatify(T)


function copy!(dest, src)
    for i in eachindex(dest,src)
        dest[i] = src[i]
    end
    dest
end
