

using MLJ,DataFrames,CSV,Random

const w=64
const h=64
const length=4096
const str="olivetti_faces"




of=olivetti_faces=CSV.File("./olivetti-face-code/olivetti_faces.csv") |> DataFrame
coerce!(of,:label=>Multiclass)
label,X=  unpack(of, ==(:label), rng=123);


"""
    load_olivetti_faces()
    返回 olivetti face  训练数据,测试数据和标签
    return (Xtrain, Xtest), (ytrain, ytest)
    train:test=0.8
"""
function load_olivetti_faces()
    (Xtrain, Xtest), (ytrain, ytest)  = partition((X, label), 0.8, multi=true,  rng=123)
    return (Xtrain, Xtest), (ytrain, ytest)
end

return  load_olivetti_faces