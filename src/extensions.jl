# extensions.jl

abstract ExtensionOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}


abstract RestrictionOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}


immutable ZeroPadding{SRC,DEST} <: ExtensionOperator{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

function apply!{T}(op::ZeroPadding, dest, src, coef_dest::Array{T}, coef_src::Array{T})
    @assert length(coef_src) == length(src)
    @assert length(coef_dest) == length(dest)

	# We do too much work here, since we put all entries of coef_dest to zero.
	# Fix later.
	fill!(coef_dest, zero(T))

	for i in eachindex(coef_src)
		coef_dest[i] = coef_src[i]
	end
end




immutable Restriction{SRC,DEST} <: RestrictionOperator{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end


function apply!(op::Restriction, dest, src, coef_dest, coef_src)
	@assert length(coef_src) == length(src)
	@assert length(coef_dest) == length(dest)

	for i in eachindex(coef_dest)
		coef_dest[i] = coef_src[i]
	end
end



