# extensions.jl

# The Extension operator is used to extend a basis of a given length to
# a basis of larger length. The resulting function should be (approximately)
# the same.
immutable Extension{SRC,DEST} <: AbstractOperator{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

# The Restriction operator is used to reduce a basis to a basis of smaller
# length. This may entail some loss of accuracy.
immutable Restriction{SRC,DEST} <: AbstractOperator{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end


# Default implementation of an extension uses zero-padding of coef_src to coef_dest
function apply!(op::Extension, dest, src, coef_dest, coef_src)

	T = eltype(coef_dest)

	# We do too much work here, since we put all entries of coef_dest to zero.
	# Fix later.
	fill!(coef_dest, zero(T))

	# And here we assume the indices of coef_dest and coef_src are the same.
	# Specialization needed if this is not the case.
	for i in eachindex(coef_src)
		coef_dest[i] = coef_src[i]
	end
end



# Default implementation of a restriction selects coef_dest from the start of coef_src
function apply!(op::Restriction, dest, src, coef_dest, coef_src)
	# Here we assume the indices of coef_dest and coef_src are the same.
	# Specialization needed if this is not the case.
	for i in eachindex(coef_dest)
		coef_dest[i] = coef_src[i]
	end
end
