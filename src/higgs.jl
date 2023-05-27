#

using Base.CoreLogging
using Logging

using Colors, ColorSchemes
using Statistics
using DataFrames
using CSV

using CairoMakie
using GLMakie

# colors for people with deficient color vision
mycolors = ColorScheme(distinguishable_colors(10, transform=protanopic))

# file loading, to adapt to your location
higgs = CSV.File("/home/jpboth/Data/HIGGS.csv", header=false)

# convert to a matrix
data = higgs|>DataFrame|>Matrix

# get correlation for last columns (first column is label boson/notboson)
# we see the role of the 7 added variables :  correlation appear
rhomat = cor(data[1:end, 2:end])
fig, ax,hm = heatmap(rhomat)
Colorbar(fig[:, end+1], hm)
fig

# Some columns takes only 3 values : columns 14, 10 and 18¶
# col 14 has 3 values, 10, 18 also)
hist(data[1:end, 14], bins = 300)
fig = CairoMakie.scatter(data[1:end, 11], data[1:end, 9], color = mycolors[data[1:end,1]], markersize=1)

#### The 2 following images show that for low values of 5th variable we get segregation of label 1 (Boson)
fig = CairoMakie.scatter(data[1:end, 4], data[1:end, 5], color = mycolors[data[1:end,1]], markersize=1)
fig = CairoMakie.scatter(data[1:end, 14], data[1:end, 5], color = mycolors[data[1:end,1]], markersize=10)

# graphic with var taking 3 values!
fig = CairoMakie.scatter(data[1:end, 14], data[1:end, 18], color = mycolors[data[1:end,1]], markersize=10)

# variable 5 show boson segregation to low values
fig = CairoMakie.scatter(data[1:end, 11], data[1:end, 5], color = mycolors[data[1:end,1]], markersize=1)
fig = CairoMakie.scatter(data[1:end, 16], data[1:end, 5], color = mycolors[data[1:end,1]], markersize=1)

# very low diagnoal segregaation
fig = CairoMakie.scatter(data[1:end, 17], data[1:end, 6], color = mycolors[data[1:end,1]], markersize=1)
fig = CairoMakie.scatter(data[1:end, 17], data[1:end, 9], color = mycolors[data[1:end,1]], markersize=1)

# segregation of variable 2 to low values
fig = CairoMakie.scatter(data[1:end, 2], data[1:end, 7], color = mycolors[data[1:end,1]], markersize=1)
fig = CairoMakie.scatter(data[1:end, 11], data[1:end, 2], color = mycolors[data[1:end,1]], markersize=1)
fig = CairoMakie.scatter(data[1:end, 9], data[1:end, 2], color = mycolors[data[1:end,1]], markersize=1)


#### variables 2 and 7 are correlated but their link to boson label is inverse 
Plots.plot(rhomat[2,1:end])
fig = CairoMakie.scatter(data[1:end, 9], data[1:end, 7], color = mycolors[data[1:end,1]], markersize=1)



#
# We create functions to get 2d random projections of centered, rescaled to unit variance of each variable
#


# center and rescale column 2 to end keeping label boson/notboson first column
function rescale(data) 
    m = mean(data[1:end, 2:end], dims=1)
    s2 = var(data[2:end], dims=1)
    #
    datan = deepcopy(data)
    datan[1:end, 2:end] = (datan[1:end, 2:end] .- m) ./ sqrt.(s2)
    return datan
end


# generate basic random value for creation of random orthogonal matrix (See Achlioptas 2003)
function randval(s)
    xsi = rand()
    if xsi < 1. / (2. * s)
        return sqrt(s)
    elseif xsi < 1. / s
        return - sqrt(s)
    else 
        return 0.
    end
end

# return sparse random orthogonal matrix (See Achlioptas 2003)
function randmatrix((nrow, ncol), s) 
    ra = zeros(ncol, 2)
    map!(x -> randval(s), ra, ra)
    return ra
end


# computes data multiplied by random projection matrix using preceding function.
# Input a (n,m) matrix, ouptut a (n,2) matrix
function proj2d(data)
    dima = size(data)
    rp = randmatrix(dima, 3.)
    return data * rp
end


# takes full higgs data (possibly centered, rescaled) with boson/not boson in first column and 
# project data from column 2 to end via random projection,
# restore boson/ notboson in first column
function higgsproj2d(data) 
    aux =  data[1:end, 2:end]
    projected = proj2d(aux)
    # restore first column?
    newdata = fill(1., (size(data)[1], 3))
    newdata[1:end, 2:3] = projected
    # transfer labels into first column
    newdata[1:end,1] = data[1:end,1]
    return newdata
end



# separate boson from nonboson from the full data matrix
function separate(data)
    nbboson = 0
    nbnotboson = 0
    for i in 1:nbrow
        if data[i,1] > 0
            nbboson += 1
        else
            nbnotboson += 1
        end
    end
    boson = Array{Float64}(undef, (nbboson,28))
    notboson = Array{Float64}(undef, (nbnotboson,28))
    nbboson = 0
    nbnotboson = 0
    for i in 1:nbrow
        if data[i,1] > 0
            nbboson += 1
            boson[nbboson, 1:end] = data[i, 2:end]
        else
            nbnotboson += 1
            notboson[nbnotboson, 1:end], data[i, 2:end]
        end
    end
    boson, notboson
end

meanboson = mean(boson, dims=1)
meannotboson = mean(notboson, dims=1)

varboson = var(boson, dims=1)
varnotboson = var(notboson, dims=1)

tboson = ( meanboson .- meannotboson) ./ sqrt.( (varboson ./ nbboson).+ (varnotboson ./ nbboson) )