
#####################
# Generic evaluation
#####################

"Compute the evaluation matrix of the given dict on the given set of points."
function evaluation_matrix(Φ::Dictionary, pts, T = codomaintype(Φ))
    A = Array{T}(undef, length(pts), length(Φ))
    evaluation_matrix!(A, Φ, pts)
end

function evaluation_matrix!(A::AbstractMatrix, Φ::Dictionary, pts)
    @assert size(A,1) == length(pts)
    @assert size(A,2) == length(Φ)

    for (j,ϕ) in enumerate(Φ), (i,x) in enumerate(pts)
        A[i,j] = ϕ(x)
    end
    A
end

@deprecate dense_evaluation default_evaluation

function default_evaluation(::Type{T}, Φ::Dictionary, gb::GridBasis; options...) where {T}
    A = evaluation_matrix(Φ, grid(gb), T)
    ArrayOperator(A, Φ, gb)
end


evaluation(Φ::Dictionary, grid::AbstractGrid; options...) =
    evaluation(coefficienttype(Φ), Φ, grid; options...)

evaluation(::Type{T}, Φ::Dictionary, grid::AbstractGrid; options...) where {T} =
    evaluation(T, Φ, GridBasis{T}(grid); options...)

evaluation(::Type{T}, src::Dictionary, dest::GridBasis; options...) where {T} =
    evaluation(T, src, dest, grid(dest); options...)

evaluation(::Type{T}, Φ::Dictionary, grid::SubGrid; options...) where {T} =
     restriction(T, supergrid(grid), grid; options...) * evaluation(T, Φ, supergrid(grid); options...)

function evaluation(::Type{T}, Φ::Dictionary, gb::GridBasis, grid; verbose=false, options...) where {T}
    verbose && println("WARN: No fast evaluation available in $grid for dictionary $Φ, using dense evaluation matrix instead.")
    default_evaluation(T, Φ, gb; verbose, options...)
end

function resize_and_transform(::Type{T}, Φ::Dictionary, gb::GridBasis, grid; verbose=false, options...) where {T}
    if size(Φ) == size(grid)
        transform_to_grid(T, Φ, gb, grid; verbose, options...)
    elseif length(grid) > length(Φ)
        dlarge = resize(Φ, size(grid))
        transform_to_grid(T, dlarge, gb, grid; verbose, options...) *
            extension(T, Φ, dlarge; verbose, options...)
    else
        verbose && println("WARN: Resize and transform: dictionary evaluated in small grid")
        default_evaluation(T, Φ, gb; verbose, options...)
    end
end
