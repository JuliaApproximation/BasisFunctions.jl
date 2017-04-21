# cardinal_b_splines.jl
module Cardinal_b_splines

export evaluate_Bspline, evaluate_periodic_Bspline, Degree

# Implementation of cardinal B splines of degree N
typealias Degree{N} Val{N}
function evaluate_symmetric_periodic_Bspline{T<:Real}(N::Int, x::Real, period::Real, ::Type{T})
  evaluate_periodic_Bspline(N, x+T(N+1)/2, period, T)
end

function periodize(x::Real, period::Real)
  x -= period*fld(x, period)
  x >= period && (x -= period)
  @assert(0<= x < period)
  x
end

function evaluate_periodic_Bspline{T<:Real}(N::Int, x::Real, period::Real, ::Type{T})
  @assert period > 0
  x = periodize(x, period)
  res = T(0)
  for k in 0:floor(Int, (N+1-x)/period)
    res += evaluate_Bspline(N, x+period*k, T)
  end
  res
end

evaluate_Bspline{T<:Real}(N::Int, x::Real, ::Type{T}) = evaluate_Bspline(Degree{N}, x, T)

function evaluate_Bspline{N}(::Type{Degree{N}}, x::Real, T::Type)
  T(x)/T(N)*evaluate_Bspline(Degree{N-1}, x, T) +
      (T(N+1)-T(x))/T(N)*evaluate_Bspline(Degree{N-1}, x-1, T)
end

evaluate_Bspline{T<:Real}(::Type{Degree{0}}, x::Real, ::Type{T}) = (0 <= x < 1) ? T(1) : T(0)

function evaluate_Bspline{T<:Real}(::Type{Degree{1}}, x::Real, ::Type{T})
  if (0 <= x < 1)
    return T(x)
  elseif (1 <= x < 2)
    return T(2) - T(x)
  else
    return T(0)
  end
end

@eval function evaluate_Bspline{T<:Real}(::Type{Degree{2}}, x::Real, ::Type{T})
  if (0 <= x < 1)
    return @evalpoly(T(x),T(0), T(0), T(1/2))
  elseif (1 <= x < 2)
    return @evalpoly(T(x),T(-3/2), T(3), T(-1))
  elseif (2 <= x < 3)
    return @evalpoly(T(x),T(9//2), T(-3), T(1//2))
  else
    return T(0)
  end
end

@eval function evaluate_Bspline{T<:Real}(::Type{Degree{3}}, x::Real, ::Type{T})
  if (0 <= x < 1)
    return @evalpoly(T(x), T(0), T(0), T(0), T(1//6))
  elseif (1 <= x < 2)
    return @evalpoly(T(x), T(2//3), T(-2), T(2), T(-1//2))
  elseif (2 <= x < 3)
    return @evalpoly(T(x), T(-22//3), T(10), T(-4), T(1//2))
  elseif (3 <= x < 4)
    return @evalpoly(T(x), T(32//3), T(-8), T(2), T(-1//6))
  else
    return T(0)
  end
end

@eval function evaluate_Bspline{T<:Real}(::Type{Degree{4}}, x::Real, ::Type{T})
  if (0 <= x < 1)
    return @evalpoly(T(x), T(0), T(0), T(0), T(0), T(1//24))
  elseif (1 <= x < 2)
    return @evalpoly(T(x), T(-5//24), T(5//6), T(-5/4), T(5//6), T(-1//6))
  elseif (2 <= x < 3)
    return @evalpoly(T(x), T(155//24), T(-25//2), T(35//4), T(-5//2), T(1//4))
  elseif (3 <= x < 4)
    return @evalpoly(T(x), T(-655//24), T(65//2), T(-55//4), T(5//2), T(-1//6))
  elseif (4 <= x < 5)
    return @evalpoly(T(x), T(625//24), -T(125//6), T(25//4), T(-5//6), T(1//24))
  else
    return T(0)
  end
end

end # module Cardinal_b_splines
