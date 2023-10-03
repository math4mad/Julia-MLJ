using  CSV,DataFrames,FileIO,GLMakie

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



marker_style=(marker=:circle,markersize=10,color=(:lightgreen,0.1),strokewidth=2,strokecolor=:black)