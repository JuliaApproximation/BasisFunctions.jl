## Pretty printing
# Methods that override the standard show(io::IO, x), to be better understandable.

## set DO_PRETTYPRINTING to false to disable pretty printing

global DO_PRETTYPRINTING = true


pretty_object_types = (
    AbstractMap,
    AbstractGrid,
    AbstractOperator,
    Dictionary,
    Measure,
)


do_pretty_printing() = Base.eval(BasisFunctions, :(DO_PRETTYPRINTING = true; nothing))
no_pretty_printing() = Base.eval(BasisFunctions, :(DO_PRETTYPRINTING = false; nothing))

if DO_PRETTYPRINTING
    for OT in pretty_object_types
        @eval begin
            function show(io::IO, object::$OT)
                if DO_PRETTYPRINTING
                    pretty_show(io, object)
                else
                    Base.show_default(io, object)
                end
            end
        end
    end
end

## Convention
#
# Each objects implements `name`, `string`, `strings`, `symbol` and `(has)stencil`.
# - name: the name of the object (defuault: the name of its type)
# - string: this will be shown in brief summaries (default: the name)
# - strings: a multiline structured representation of the object (default defined in terms of string)
# - symbol: a preferred symbol to use in condensed structure representation
# - hasstencil: the object has a condensed structured representation
# - stencil: the structure representation of the object in terms of symbols

# If an object has a stencil we show that, otherwise we show a "simple" string
pretty_show(io, object) =
    hasstencil(object) ? show_composite(io, object) : show_simple(io, object)

# The "simple" representation of an object by default prints its "strings"
show_simple(io::IO, object) = println(io, print_strings(strings(object), 0, ""))


## Default names

for OT in pretty_object_types
    # The default name is the name of the type
    @eval name(object::$OT) = string(typeof(object).name)
    # The default string is the name of the object
    @eval string(object::$OT) = name(object)
    # By default, the object has no stencil, unless it is a composite object
    @eval hasstencil(object::$OT) = iscomposite(object)
end

"Does the object require parentheses around the composite elements of its stencil?"
stencil_parentheses(object) = false

"Does the object require parentheses if part of a composite stencil?"
object_parentheses(object) = false


####
# Operator symbols and strings
####

# The default strings is just a tuple of the string
strings(op::AbstractOperator) = (string(op),)

# Complex expressions substitute strings for symbols.
# Default symbol is first letter of the string
symbol(op::AbstractOperator) = string(op)[1]

# Common symbols (to be moved to respective files)
string(op::MultiplicationOperator) = string(op, op.object)
string(op::MultiplicationOperator, object) = "Multiplication by " * string(typeof(op.object))

symbol(op::MultiplicationOperator) = symbol(op, op.object)
symbol(op::MultiplicationOperator, object) = "M"

symbol(op::MultiplicationOperator, object::FFTW.cFFTWPlan{T,K}) where {T,K} = K<0 ? "FFT" : "iFFT"
symbol(op::MultiplicationOperator, object::FFTW.DCTPlan{T,K}) where {T,K} = K==FFTW.REDFT10 ? "DCT" : "iDCT"

function string(op::MultiplicationOperator, object::FFTW.cFFTWPlan)
    io = IOBuffer()
    print(io,op.object)
    match(r"(.*?)(?=\n)",String(take!(io))).match
end

function string(op::MultiplicationOperator, object::FFTW.DCTPlan)
    io = IOBuffer()
    print(io,op.object)
    String(take!(io))
end

function string(v::AbstractVector)
    io = IOBuffer()
    if length(v) > 6
       inds = axes(v,1)
       Base.show_delim_array(io, v, "[", ",", "", false, inds[1], inds[1] + 2)
       print(io, "  …  ")
       Base.show_delim_array(io, v, "", ",", "]", false, inds[end - 2], inds[end])
    else
       Base.show_delim_array(io, v, "[", ",", "]", false)
    end
    String(take!(io))
end

strings(op::DiagonalOperator) = ("Diagonal operator with element type $(eltype(op))", strings(diag(op)))


# Different operators with the same symbol get added subscripts
subscript(i::Integer) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))




####
# Dictionary symbols and strings
####


symbol(dict::Dictionary) = string(dict)[1]

# Default string is a brief structured representation of the dictionary
strings(d::Dictionary) = (string(d),
    ("length = $(length(d))",
     "$(domaintype(d)) -> $(codomaintype(d))",
     "support = $(support(d))"))




####
# Grid symbols and strings
####


symbol(g::AbstractGrid) = "g"

strings(g::AbstractGrid) = (name(g) * " of size $(size(g)),\tELT = $(eltype(g))",)



####
# Measure symbols and strings
####

symbol(m::Measure) = "μ"

strings(m::Measure) = (name(m), (string(support(m)),))



####
# Map symbols and strings
####

# This definition is missing from DomainSets
iscomposite(m::AbstractMap) = false

symbol(m::AbstractMap) = "M"

strings(m::AbstractMap) = (string(m),)

strings(m::AffineMap) = (string("Affine map y = ", m.a, " * x + ", m.b),)


#### Actual printing methods.

# extend children method from AbstractTrees

for OT in pretty_object_types
    @eval children(object::$OT) = iscomposite(object) ? elements(object) : ()
end

function symbol_leaves(object)
    if !hasstencil(object)
        return [object]
    else
        A = stencilarray(object)
        S = Any[]
        for a in A
            if !(a isa String || a isa Char)
                push!(S, symbol_leaves(a)...)
            end
        end
        return S
    end
end

# Collect all symbols used
function symbollist(op)
    # Find all leaves
    ops = symbol_leaves(op)
    # find the unique elements in ops
    U = unique(ops)
    # Create a dictionary that maps the unique elements to their Symbols
    S = Dict{Any,Any}(u => symbol(u) for u in U)
    # Get a list of the unique symbols
    Sym = unique(values(S))
    for i in 1:length(Sym)
        # For each of the unique symbols, filter the dictionary for those values
        Sf = filter(u->u.second==Sym[i],S)
        if length(Sf)>1
            j=1
            for k in keys(Sf)
                    S[k]=S[k]*subscript(j)
                j+=1
            end
        end
    end

    # We iterate over the tree, leaves first, and try to identify large substencils
    It = PostOrderDFS(op)
    j = 1
    Pops = Any[]
    for Pop in It
        if hasstencil(Pop) && (in(Pop,Pops) || (nchildren(Pop)>5 && nchildren(Pop)<nchildren(op)/2))
            S[Pop] = "Ψ"*subscript(j)
            j += 1
        end
        hasstencil(Pop) && nchildren(Pop)>5 && push!(Pops,Pop)
    end
    S
end

# Compute the stencil of the object recursively. We invoke stencilarray recursively.
# However, if we encounter an object that already has a symbol in S, we stop the
# recursion and just add the symbol. (This happens if a complicated object was
# awarded a substencil).
function recursive_stencil(object, S)
    if haskey(S, object)
        return [S[object]]
    end
    A = stencilarray(object)
    i = 1
    k = length(A)
    while i <= k
        a = A[i]
        if !(a isa String || a isa Char) && hasstencil(a)
            Asub = recursive_stencil(a, S)
            if stencil_parentheses(object) && object_parentheses(a)
                splice!(A, i+1:i, ")")
                splice!(A, i:i, Asub)
                splice!(A, i:(i-1), "(")
            else
                splice!(A, i:i, Asub)
            end
        end
        i += 1
        k = length(A)
    end
    A
end

# When printing a stencil, replace all operators/dictionaries with their symbol.
# Strings are printed directly
function printstencil(io, object, S)
    A = recursive_stencil(object, S)
    for a in A
        if haskey(S, a)
            print(io, S[a])
        else
            print(io, a)
        end
    end
end

function show_composite(io::IO, op::AbstractOperator)
    print(io, "Operator ")
    show_composite_object(io, op)
end

function show_composite(io::IO, dict::Dictionary)
    print(io, "Dictionary ")
    show_composite_object(io, dict)
end

function show_composite(io::IO, measure::Measure)
    print(io, "Measure ")
    show_composite_object(io, measure)
end

function show_composite(io::IO, g::AbstractGrid)
    print(io, "Grid ")
    show_composite_object(io, g)
end

function show_composite(io::IO, map::AbstractMap)
    print(io, "Map ")
    show_composite_object(io, g)
end


# Main printing method, first print the stencil and any remaining composites,
# then show a full list of symbols and their strings.
function show_composite_object(io::IO, object)
    S = symbollist(object)
    printstencil(io, object, S)
    print(io, "\n\n")
    SortS = sort(collect(S), by=x->string(x[2]), rev=true)
    for (key,value) in SortS
        if iscomposite(key)
            print(io, value, " = ")
            delete!(S, key)
            printstencil(io, key, S)
            print(io, "\n\n")
        end
    end
    SortS = sort(collect(S), by=x->string(x[2]), rev=true)
    for (key,value) in SortS
        print(io, value, "\t:\t", print_strings(strings(key),0,"\t\t"))
    end
end


strings(any) = tuple(repr(any))
strings(vc::AbstractVector) = (string(vc),)

# Determine number of children of operator (to determine where to split)
function nchildren(op)
    It = PostOrderDFS(op)
    j=0
    for i in It
        j+=1
    end
    j
end

# These functions convert the strings tuples to multiline strings, by prefixing a variable number of spaces, and possibly a downrright arrow
function print_strings(strings::Tuple, depth=0, prefix="")
    s = strings
    result = ""
    for s in strings
        result *= print_strings(s, depth+1, prefix*"  ")
    end
    result
end

function print_strings(strings::AbstractString, depth=0, prefix="")
    if depth == 1
        result = strings*"\n"
    else
        result = prefix*"↳ "*strings*"\n"
    end
    result
end
