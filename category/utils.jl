using  CSV,DataFrames,FileIO,GLMakie,MLJ

function load_csv(str::AbstractString)::AbstractDataFrame
    df = str |> d -> CSV.File("../dataset/$str.csv") |> DataFrame |> dropmissing
    return df
end


function boundary_data(df::AbstractDataFrame;n=200)
    n1=n2=n
    xlow,xhigh=extrema(df[:,:x])
    ylow,yhigh=extrema(df[:,:y])
    tx = range(xlow,xhigh; length=n1)
    ty = range(ylow,yhigh; length=n2)
    x_test = mapreduce(collect, hcat, Iterators.product(tx, ty));
    xtest=MLJ.table(x_test',names=[:x,:y])
    return tx,ty,xtest
end


function load_german_creditcard()
    to_ScienceType(d)=coerce(d,autotype(d, (:few_to_finite, :discrete_to_continuous)))
    df=CSV.File("../dataset/german_creditcard.csv") |> DataFrame|>to_ScienceType
    y, X=  unpack(df, ==(:Creditability));
    cat=levels(y)
    X=X[:,1:end-1]  # 去除最后一列变量
    (Xtrain, Xtest), (ytrain, ytest) = partition((X, y), 0.8, rng=123, multi=true);
    return Xtrain, Xtest, ytrain, ytest,cat
end


marker_style=(marker=:circle,markersize=10,color=(:lightgreen,0.1),strokewidth=2,strokecolor=:black)


load_german_creditcard()