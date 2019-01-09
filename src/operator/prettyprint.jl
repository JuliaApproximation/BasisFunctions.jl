## OPERATORS

# Methods that override the standard show(io::IO,op::AbstractOperator), to be better understandable.

####
# set DO_PRETTYPRINTING to false to disable pretty printing
####
# Delegate to show_operator
global DO_PRETTYPRINTING = true
if DO_PRETTYPRINTING
    show(io::IO, op::AbstractOperator) = (has_stencil(op)) ? show_composite(io,op) : show_operator(io, op)
    show(io::IO,s::Span) = show(io,dictionary(s))
    show(io::IO, d::Dictionary) = (has_stencil(d)) ? show_composite(io,d) : show_dictionary(io, d)
end


####
# Operator symbols and strings
####


# Default is the operator string
show_operator(io::IO,op::AbstractOperator) = println(print_strings(strings(op),0,""))
# show_operator(io::IO,op::AbstractOperator) = println(string(op))

# Default string is the string of the type
# string(op::DictionaryOperator) = match(r"(?<=\.)(.*?)(?=\{)",string(typeof(op))).match
string(op::DictionaryOperator) = string(typeof(op))
string(op::AbstractOperator) = string(typeof(op))

# Complex expressions substitute strings for symbols.
# Default symbol is first letter of the string
symbol(op::AbstractOperator) = string(op)[1]

# Common symbols (to be moved to respective files)
symbol(E::IndexRestrictionOperator) = "R"
symbol(E::IndexExtensionOperator) = "E"

string(op::MultiplicationOperator) = string(op,op.object)
string(op::MultiplicationOperator,object) = "Multiplication by "*string(typeof(op.object))

symbol(op::MultiplicationOperator) = symbol(op,op.object)
symbol(op::MultiplicationOperator,object) = "M"

symbol(op::MultiplicationOperator,object::FFTW.cFFTWPlan{T,K}) where {T,K} = K<0 ? "FFT" : "iFFT"
symbol(op::MultiplicationOperator,object::FFTW.DCTPlan{T,K}) where {T,K} = K==FFTW.REDFT10 ? "DCT" : "iDCT"

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

function strings(a::AbstractVector)
    if length(a) > 8
        tuple(repr(a[1:8]) * "...")
    else
        tuple(repr(a))
    end
end

strings(op::AbstractOperator) = tuple(string(op))

strings(op::DiagonalOperator) = ("Diagonal operator with element type $(eltype(op))", strings(diagonal(op)))


# Different operators with the same symbol get added subscripts
subscript(i::Integer) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))

strings(m::Measure) = tuple(string(m))
string(m::FourierMeasure{T}) where {T} = "The Lebesgue measure on [0,1] (T = $T)"
string(m::ChebyshevTMeasure{T}) where {T} = "The Chebyshev measure of the first kind on [-1,1] (T = $T)"
string(m::ChebyshevUMeasure{T}) where {T} = "The Chebyshev measure of the second kind on [-1,1] (T = $T)"


####
# Parentheses for operators
####

# Include parentheses based on precedence rules
# By default, don't add parentheses
parentheses(t::AbstractOperator,a::AbstractOperator) = false
# Sums inside everything need parentheses
parentheses(t::CompositeOperators,a::OperatorSum) = true
parentheses(t::TensorProductOperator,a::OperatorSum) = true
# Mixing and matching products need parentheses
parentheses(t::CompositeOperators,a::TensorProductOperator) = true
parentheses(t::TensorProductOperator,a::CompositeOperators) = true
parentheses(d::OperatedDict,a::CompositeOperators) = true

####
# Dictionary symbols and strings
####

# Default is the operator string
show_dictionary(io::IO,d::Dictionary) = println(print_strings(strings(d),0,""))

# Default string is the string of the type
strings(d::Dictionary) = (name(d),("length = $(length(d))","$(domaintype(d)) -> $(codomaintype(d))","support = $(support(d))"))
strings(d::GridBasis) = ("A grid basis for coefficient type $(coefficienttype(d))",strings(grid(d)))
strings(g::AbstractGrid) = (name(g)*" of size $(size(g)),\tELT = $(eltype(g))",)
strings(d::DerivedDict) = (name(d),)

symbol(d::Dictionary) = name(d)[1]

## Default names
name(d::Dictionary) = _name(d)
name(g::AbstractGrid) = _name(g)
name(o::AbstractOperator) = _name(o)

function _name(anything)
    m = match(r"(?<=\.)(.*?)(?=\{)",string(typeof(anything)))
    if m == nothing
        m2 = match(r"(.*?)(?=\{)",string(typeof(anything)))
        if m2 == nothing
            string(typeof(anything))
        else
            String(m2.match)
        end
    else
        String(m.match)
    end
end


####
# Dictionary Parentheses
####


parentheses(t::Dictionary, d::Dictionary) = false
parentheses(t::CompositeDict, a::TensorProductDict)=true



has_stencil(anything) = is_composite(anything)
#### Actual printing methods.

# extend children method from AbstractTrees
children(A::Union{Dictionary,AbstractOperator}) = is_composite(A) ? elements(A) : ()
function myLeaves(op::BasisFunctions.DerivedOperator)
    A = Any[]
    push!(A,op)
    push!(A,myLeaves(superoperator(op))...)
    return A
end
function myLeaves(op::BasisFunctions.DerivedDict)
    A = Any[]
    push!(A,op)
    push!(A,myLeaves(superdict(op))...)
    return A
end

function myLeaves(op)
    A = Any[]
    if !has_stencil(op)
        push!(A,op)
    else
        for child in BasisFunctions.children(op)
            push!(A,myLeaves(child)...)
        end
    end
    return A
end

# Collect all symbols used
function symbollist(op)
    # Find all leaves
    ops = myLeaves(op)
    # find the unique elements in ops
    U = unique(ops)
    # Create a dictionary that maps the unique elements to their Symbols
    S = Dict{Any,Any}(U[i] => symbol(U[i]) for i=1:length(U))
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
    It = PostOrderDFS(op)
    j=1
    Pops = Any[]
    for Pop in It
        if has_stencil(Pop) && (in(Pop,Pops) || (nchildren(Pop)>5 && nchildren(Pop)<nchildren(op)/2))
            S[Pop] = "Ψ"*subscript(j)
            j+=1
        end
        has_stencil(Pop) && nchildren(Pop)>5 && push!(Pops,Pop)
    end
    S
end

    # Stencils define the way ParentOperators are printed (to be moved to proper files)
stencil(op)=op

function stencil(op,S)
    if !has_stencil(op) && haskey(S,op)
        return op
    else
        A=stencil(op)
        return recurse_stencil(op,A,S)
    end
end

# Any remaining operator/dictionary that has a stencil will be
function recurse_stencil(op,A,S)
    i=1
    k=length(A)
    while i<=k
        if !(typeof(A[i])<:String || typeof(A[i])<:Char) && has_stencil(A[i])
            if parentheses(op,A[i])
                splice!(A,i+1:i,")")
                splice!(A,i:i,stencil(A[i],S))
                 splice!(A,i:(i-1),"(")
            else
                splice!(A,i:i,stencil(A[i],S))
            end
        end
        i+=1
        k=length(A)
    end
    A
end

# When printing a stencil, replace all operators/dictionaries with their symbol. Strings are printed directly
function printstencil(io,op,S)
    A = stencil(op,S)
    for i = 1:length(A)
        if haskey(S,A[i])
            print(io,S[A[i]])
        else
            print(io,A[i])
        end
    end
end


# Main printing method, first print the stencil and any remaining composites, then show a full list of symbols and their strings.
function show_composite(io::IO,op)
    S = symbollist(op)
    printstencil(io,op,S)
    print(io,"\n\n")
    SortS=sort(collect(S),by=x->string(x[2]),rev=true)
    for (key,value) in SortS
        if is_composite(key) && !isa(key,DerivedOperator) && !isa(key,DerivedDict)
            print(io,value," = ")
            delete!(S,key)
            printstencil(io,key,S)
            print(io,"\n\n")
        end
    end
    SortS=sort(collect(S),by=x->string(x[2]),rev=true)
    for (key,value) in SortS
        print(io,value,"\t:\t",print_strings(strings(key),0,"\t\t"))
    end
end


strings(any) = tuple(repr(any))

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
    for i in 1:length(strings)
        result = result*print_strings(s[i], depth+1, prefix*"  ")
    end
    result
end

function print_strings(strings::AbstractString, depth=0,prefix="")
    if depth == 1
        result = strings*"\n"
    else
        result = prefix*"↳ "*strings*"\n"
    end
    result
end
