# using WaveletsCopy: evaluate_in_dyadic_points, evaluate_periodic_in_dyadic_points
# using WaveletsCopy.DWT: periodic_support
# """
# Return the quadrature coefficients for an integration method that
# integrates polynomials up to degree M exactly, multiplied by a scaling
# function phi defined by lo_d.
# The coefficients are calculated using chebyshev polynomials for
# stability. The grid points are chosen equidistant in the support of
# the scaling function.
# cfr. W. Sweldens, "The construction and application of wavelets in
# numerical analysis".
# """
# function quad_sf_weights(lo_d::AbstractVector{ELT}, M::Int; cutoff=1e-12, options...) where ELT
#     K = (length(lo_d)-1)/2
#
#     A = vdm_matrix_chebyshev(linspace(-BigFloat(1),BigFloat(1),M+1))
#
#     m = sf_chebyshev_moments(BigFloat.(lo_d), M)
#
#     w = A\m
#     ELT.(w)[:]
# end
#
# """"
# Return the uniformly distributed points that are used in the quadrature
# rule of the given scaling function.
# """
# function quad_sf_points(lo_f::AbstractVector{ELT}, M::Int) where ELT
#     K = (length(lo_f)-1)//2
#     linspace(ELT(-K), ELT(K), M+1)
# end
#
# function vdm_matrix_chebyshev(t::AbstractVector{ELT}) where {ELT}
#     M = length(t)-1;
#     A = zeros(M+1,M+1);
#
#     for i in 0:M
#         A[i+1,:] = chebyshev_eval.(i, t);
#     end
#
#     A
# end
#
# "Evaluate the chebyshev polynomial of degree p in x."
# chebyshev_eval(p, x) = real(cos(p*acos(x)))
#
#
# """
# Calculate the modified moments of the scaling function defined by lo_d.
# """
# function sf_chebyshev_moments(lo_f::AbstractVector{ELT}, M::Int) where {ELT}
#     m = zeros(ELT,M+1,1)
#     m[1] = 1
#     K = (length(lo_f)-1)/2
#
#     w = zeros(ELT,M+1,M+1,Int(2*K+1))
#     for k in -K:K
#         w[:,:,Int(k+K+1)] = local_chebyshev_w(-K, K, k, M)
#     end
#
#     for p in 1:M
#         d = 0
#         for i in 0:p-1
#             e = 0
#             for k in -K:K
#                 # % CAVE: our coefficients differ by a factor 1/sqrt(2) from Sweldens!
#                 e = e + ELT(1)/sqrt(ELT(2))*lo_f[Int(k+K+1)]*w[p+1,i+1,Int(k+K+1)]
#             end
#             d = d + e*m[i+1]
#         end
#         m[p+1] = 1/(2^p-1)*d
#     end
#
#     m
# end
#
# """
# Calculate the coefficients w in the algorithm by Sweldens and Piessens.
# """
# function local_chebyshev_w(a::ELT, b::ELT, k, M::Int) where ELT
#     L = b-a;
#     lambda = ELT(2)*(k-a)/L-1
#
#     w = zeros(ELT,M+1,M+1)
#
#     w[1,1] = 1
#     w[2,1] = lambda
#     w[2,2] = 1
#     if M > 1
#         w[3,1] = 2*lambda^2-3
#         w[3,2] = 4*lambda
#         w[3,3] = 1
#     end
#
#     # p+1 is the degree of the next polynomial
#     for p in 2:M-1
#         w[p+2,1] = w[p+1,2] + 2*lambda*w[p+1,1] - 4*w[p,1]
#         w[p+2,2] = 2*w[p+1,1]+w[p+1,3]+2*lambda*w[p+1,2]-4*w[p,2]
#         for i in 2:p-1
#             w[p+2,i+1] = w[p+1,i]+w[p+1,i+2]+2*lambda*w[p+1,i+1]-4*w[p,i+1]
#         end
#         w[p+2,p+1] = w[p+1,p] + 2*lambda*w[p+1,p+1]
#         w[p+2,p+2] = w[p+1,p+1]
#     end
#
#     w
# end
#
# """
# The weights of the quadrature ∑w_if(x_i) ≈ ∫f(x)φ(x)dx where ∫ϕ(x)dx = 1 and ϕ is the scaling function.
# """
# quad_sf_weights(wav::DiscreteWavelet, M::Int; options...) =
#     quad_sf_weights(filter(Primal, scaling, wav).a, M; options...)
#
# quad_sf_weights(side::Side, kind::Kind, wav::DiscreteWavelet, M::Int; options...) =
#     quad_sf_weights(filter(side, kind, wav).a, M; options...)
#
#
#
#
# fun_data(f::Function, side::Side, kind::Kind, wav::DiscreteWavelet, M::Int, j::Int, k::Int, d::Int) =
#     y = f.(linspace(support(side, kind, wav, j, k)..., (1<<d)*M+1))
#
# function fun_data_per(f::Function, side::Side, kind::Kind, wav::DiscreteWavelet, M::Int, j::Int, k::Int, d::Int)
#     intervals = periodic_support(side, kind, wav, j, k)
#     L = support_length(side, kind, wav)
#     s = Int(M/L)
#     dx = 1/(1<<(j+d))/s
#     t = intervals[1]
#     x = collect(linspace(t[1], t[2], Int((t[2] - t[1])/dx+1)))
#     for i in 2:length(intervals)
#         t = intervals[i]
#         x = vcat(x, collect(linspace(t[1], t[2], Int((t[2] - t[1])/dx+1))[2:end]) )
#     end
#     # info("Length is $(length(x))")
#     y = f.(x)
# end
#
# """
# Coefficients c_r = ∫f(x)ϕ_{j+d,r}(x)dx using M*2^d evaluation points, r=some array depending on k, j, d.
# """
# function lowest_scale(y::Vector, side::Side, kind::Kind, wavelet::DiscreteWavelet{ELT}, M::Int, j::Int, k::Int, d::Int) where {ELT}
#     w = quad_sf_weights(side, kind, wavelet, M)
#     L = support_length(side, kind, wavelet)
#     s = Int(M/L)
#     a = zeros(ELT,1+(1<<d-1)*L)
#     for i in 1:1+(1<<d-1)*L
#         a[i] = dot(w,y[(0:M)+1+(i-1)*s])
#     end
#     CompactSequence(a)
# end
#
# """
# DWT step ν_{j-1,l} = √2∑_k h_{k-2l}ν_{j,k} for some l's
# """
# function Base.step(L::Int, h::Sequence, x::Sequence)
#     ELT = eltype(h)
#     y = zeros(ELT,L)
#     for l in 0:L-1
#         t = 0
#         for k in firstindex(h):lastindex(h)
#             t += h[k]*x[k+2l]
#         end
#         y[l+1] = t
#     end
#     CompactSequence(y/sqrt(2))
# end
#
# """
# ∫f(x)ϕ_{j,k}(x)dx using M*2^d evaluations and a quadrature rule of M points.
# """
# quad_sf(fun::Function, wav::DiscreteWavelet, M::Int, j::Int, k::Int, d::Int=0;
#             periodic=false) =
#     quad_sf(fun, Primal, scaling, wav, M, j, k, d; periodic=periodic)
#
# function quad_sf(fun::Function, side::Side, kind::Kind, wav::DiscreteWavelet, M::Int, j::Int, k::Int, d::Int=0;
#         periodic=false)
#     y = periodic ? fun_data_per(fun, side, kind, wav, M, j, k, d):
#                    fun_data(fun, side, kind, wav, M, j, k, d)
#     quad_sf(y, side, kind, wav, M, j, k, d)
# end
#
# function quad_sf(y::Vector, side::Side, kind::Kind, wav::DiscreteWavelet, M::Int, j::Int, k::Int, d::Int=0)
#     x = lowest_scale(y, side, kind, wav, M, j, k, d)
#     L = support_length(side, kind, wav)
#     h = filter(side, kind, wav)
#     h = Sequences.shift(h, -Sequences.offset(h))
#     # println(Sequences.offset(h))
#     for i in d:-1:1
#         x = step(1+(1<<(i+j)-1)*L, h, x)
#     end
#     x[0]
# end
#
# """
# Trapezoidal rule for ∫f(x)ϕ_{j,k}(x)dx.
# """
# quad_trap(fun::Function, wav::DiscreteWavelet, j::Int, k::Int, d::Int; periodic=false) =
#     quad_trap(fun, Primal, scaling, wav, j, k, d; periodic=periodic)
#
#
# function quad_trap(fun::Function, side::Side, kind::Kind, wav::DiscreteWavelet, j::Int, k::Int, d::Int; periodic=false)
#     w, x = periodic? evaluate_periodic_in_dyadic_points(side, kind, wav, j, k, d; points=true):
#                      evaluate_in_dyadic_points(side, kind, wav, j, k, d; points=true)
#     sum(fun.(x).*w)/(1<<d)*sqrt(2<<j)/sqrt(2)
# end
#
# lowest_scale_per_N(f, wav::DiscreteWavelet, M::Int, j::Int, d::Int) =
#     lowest_scale_per_N(f, Primal, scaling, wav, M, j, d)
#
# function lowest_scale_per_N(f, side::Side, kind::Kind, wav::DiscreteWavelet{ELT}, M::Int, j::Int, d::Int) where ELT
#     L = support_length(side, kind, wav)
#     s = Int(M/L)
#     x = linspace(ELT(0), ELT(1), s*1<<(j+d)+1)[1:end-1]
#     lowest_scale_per_N(f.(x), side, kind, wav, M, j, d)
# end
#
# function lowest_scale_per_N(y::Vector, side::Side, kind::Kind, wav::DiscreteWavelet{ELT}, M::Int, j::Int, d::Int) where {ELT}
#     w = quad_sf_weights(side, kind, wav, M)
#     a = zeros(ELT, length(y))
#     L = support_length(side, kind, wav)
#     s = Int(M/L)
#     for i in 1:length(y)
#         a[i] = dot(w, y[map(x->mod(x-1,length(y))+1,(0:M)+1+(i-1)*s)])
#     end
#     PeriodicExtension(a)
# end
#
# step_N(L, h, x) = PeriodicExtension(step(L, h, x).a)
#
#
# function fun_data_N(f, side::Side, kind::Kind, wav::DiscreteWavelet{ELT}, M::Int, j::Int, d::Int) where ELT
#     L = support_length(side, kind, wav)
#     s = Int(M/L)
#     x = linspace(ELT(0), ELT(1), s*1<<(j+d)+1)[1:end-1]
#     # info("Length is $(length(x))")
#     f.(x)
# end
#
# """
# All coefficients ∫f(x)ϕ_{j,k}(x)dx for k = 0,...,2^j-1. Using multiple point quadrature.
# """
# quad_sf_N(fun::Function, wav::DiscreteWavelet, M::Int, j::Int, d::Int) =
#     quad_sf_N(fun, Primal, scaling, wav, M, j, d)
#
#
# quad_sf_N(fun::Function, side::Side, kind::Kind, wav::DiscreteWavelet, M::Int, j::Int, d::Int) =
#     quad_sf_N(fun_data_N(fun, side, kind, wav, M, j, d), side, kind, wav, M, j, d)
#
# function quad_sf_N(y::Vector, side::Side, kind::Kind, wav::DiscreteWavelet, M::Int, j::Int, d::Int)
#     x = lowest_scale_per_N(y, side, kind, wav, M, j, d)
#     L = support_length(side, kind, wav)
#     for i in d-1:-1:0
#         x = step_N(1<<(i+j),filter(side, kind, wav), x)
#     end
#     x.a
# end
#
# """
# All coefficients ∫f(x)ϕ_{j,k}(x)dx for k = 0,...,2^j-1. Using trapezoidel rule.
# """
# quad_trap_N(fun::Function, wav::DiscreteWavelet, j::Int, d::Int) =
#     quad_trap_N(fun, Primal, scaling, wav, j, d)
#
# function quad_trap_N(fun::Function, side::Side, kind::Kind, wav::DiscreteWavelet, j::Int, d::Int)
#     w, x = evaluate_periodic_in_dyadic_points(side, kind, wav, j, 0, d; points=true)
#     y = fun.(x)
#     quad_trap_N(y, w, j, d)
# end
#
# function quad_trap_N(y::Vector{ELT}, w::Vector{ELT}, j::Int, d::Int) where ELT
#     z = similar(y)
#     a = zeros(ELT, 1<<j)
#     for i in 1:1<<j
#         a[i] = dot(w, y)/(1<<d)*sqrt(2<<j)/sqrt(2)
#         circshift!(z, y,-(1<<(d-j)))
#         copy!(y, z)
#     end
#     a
# end
#
# # function quad_sf_weights(side::Side, kind::Kind, wav::DiscreteWavelet{ELT}, M::Int, j::Int, d::Int) where {ELT}
# #     L = Int(M/support_length(side, kind, wav))*support_length(side, kind, wav)*1<<d+1
# #     e = zeros(ELT, L)
# #     w = zeros(ELT, L)
# #     for i in 1:L
# #         e[i] = 1
# #         w[i] = quad_sf(e, side, kind, wav, M, j, 0, d)
# #         e[i] = 0
# #     end
# #     w
# # end
#
# function quad_sf_weights(side::Side, kind::Kind, wav::DiscreteWavelet{ELT}, M::Int, d::Int) where {ELT}
#     w = CompactSequence(quad_sf_weights(side, kind, wav, M))
#     h = filter(side, kind, wav)
#     h = Sequences.shift(h, -Sequences.offset(h))
#     s = Int(log2(Int(M/support_length(side, kind, wav))))
#     for i in 1:d
#         w = shifted_conv(h, w, 1<<(i-1+s))/sqrt(ELT(2))
#     end
#     w.a
# end
