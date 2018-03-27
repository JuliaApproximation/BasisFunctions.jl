"""
A ParentOperator is an operator that has children, i.\e.\ combines the actions of sub-operators
"""
abstract type ParentOperator{T} <: AbstractOperator{T}
end

# extend children method from AbstractTrees
children(op::AbstractOperator) = ()
children(op::ParentOperator) = elements(op)


# Collect all symbols used
function symbollist(op::ParentOperator)
    # Find all leaves
    ops = collect(Leaves(op))
    # find the unique elements in ops
    U = unique(ops)
    # Create a dictionary that maps the unique elements to their Symbols
    S = Dict{Any,Any}(U[i] => symbol(U[i]) for i=1:length(U))
    # Get a list of the unique symbols
    Sym = unique(values(S))
    for i=1:length(Sym)
        # For each of the unique symbols, filter the dictionary for those values
        Sf=filter((u,v)->v==Sym[i],S)
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
        if isa(Pop,ParentOperator) && (in(Pop,Pops) || (nchildren(Pop)>5 && nchildren(Pop)<nchildren(op)/2))
            S[Pop] = "Ψ"*subscript(j)
            j+=1
        end
        isa(Pop,ParentOperator) && nchildren(Pop)>5 && push!(Pops,Pop)
    end
    S
end
        
    # Stencils define the way ParentOperators are printed (to be moved to proper files)
stencil(op::GenericOperator)=op

function stencil(op::GenericOperator,S)
    if haskey(S,op)
        return op
    else
        A=stencil(op)
        return recurse_stencil(op,A,S)
    end
end
    
function recurse_stencil(op::GenericOperator,A,S)
    i=1
    k=length(A)
    while i<=k
        if isa(A[i],ParentOperator) && !haskey(S,A[i])
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

function printstencil(io,op,S)
    A = stencil(op,S)
    for i = 1:length(A)
        if isa(A[i],GenericOperator)
            print(io,S[A[i]])
        else
            print(io,A[i])
        end
    end
end
        

function show_operator(io::IO,op::ParentOperator)
    S = symbollist(op)
    printstencil(io,op,S)
    print(io,"\n\n")
    SortS=sort(collect(S),by=x->string(x[2]),rev=true)   
    for (key,value) in SortS
        if isa(key,ParentOperator)
            print(io,value," = ")
            delete!(S,key)
            printstencil(io,key,S)
            print(io,"\n\n")
        end
    end
    SortS=sort(collect(S),by=x->string(x[2]),rev=true)   
    for (key,value) in SortS
        println(io,value,"\t:\t",string(key))
    end
end

function nchildren(op::ParentOperator)
    It = PostOrderDFS(op)
    j=0
    for i in It
        j+=1
    end
    j
end
