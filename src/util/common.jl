# common.jl

macro add_properties(T, props...)
    e = quote end
    for prop in props
        f = quote $(esc(prop))(a::$(esc(T))) = true end
        append!(e.args, f.args)
    end
    e
end

tolerance{T}(::Type{T}) = sqrt(eps(T))
tolerance{T <: Real}(::Type{Complex{T}}) = tolerance(T)


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
    for i in eachindex(dest, src)
        dest[i] = src[i]
    end
    dest
end


#An efficient way to access elements of a Tuple type using index j
@generated function tuple_index{T <: Tuple}(::Type{T}, j)
    :($T.parameters[j])
end

@generated function tuple_length{T <: Tuple}(::Type{T})
    :($length(T.parameters))
end

isdyadic(n::Int) = n == 1<<round(Int, log2(n))
