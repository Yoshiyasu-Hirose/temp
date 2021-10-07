using LinearAlgebra 
using Statistics 
using CSV, DataFrames 
using HDF5 


function mainDA(ftime) 
     
    # initialize parameters 
    for line in eachline("input.csv") 
        words = split(line, ",", keepempty=false) 
        name = words[1] 
        if name == "NCELLS" 
            global nCells=parse(Int64, split(line, ",")[2]) 
        end  
        if name == "NUMD" 
            global member = parse(Int64, words[2]) 
        end 
        if name == "p" 
            global p = parse(Int64, words[2]) 
        end 
        if name == "N" 
            global N = parse(Int64, words[2]) 
        end 
        if name == "OBSFLAG" 
            global obsFlag = parse.(Int64, words[2:size(words,1)]) 
        end 
        if name == "NISO" 
            global osbave = parse(Float64, words[3]) 
            global obsvar = parse(Float64, words[4]) 
        end 
        if name == "NOBS" 
            global nObs = parse(Int64, words[2]) 
        end 
        if name == "INFLATION" 
            global alpha = parse(Float64, words[2]) 
        end 
        if name == "LOCALIZATION" 
            global D = parse(Float64, words[2]) 
        end 
    end   
     
    U1, U2, U3 = zeros(Float64, nCells), zeros(Float64, nCells), zeros(Float64, nCells) 
    T =  zeros(Float64, nCells) 
    Xf = zeros(Float64, N, member) 
    Xa = zeros(Float64, N, member) 
    Xf_m = zeros(Float64, N) 
    Xa_m = zeros(Float64, N) 
    dXf = zeros(Float64, N, member) 
    dXa = zeros(Float64, N, member) 
    dY = zeros(Float64, p, member) 
    K = zeros(Float64, N, p) 
    Tr = zeros(Float64, member, member) 
    dX = zeros(Float64, N, 1) 
    Pf = zeros(Float64, N, N) 
    rho_mat = zeros(Float64, N, N) 
     
    E = Matrix{Float64}(I, member, member) 
    R = Matrix{Float64}(obsvar^2*I, p, p)  
    H = zeros(Float64, p, N)  
    y = zeros(Float64, p, 1) 
     
    # temporary matrix 
    w1 = zeros(N, p) 
    w1_1 = zeros(N, p) 
    w2 = zeros(p, p) 
    w2_1 = zeros(p, p) 
    w3 = zeros(N, N) 
    w4 = zeros(N, 1) 
    w5 = zeros(p, 1) 
    w5_1 = zeros(Int64, p, 1) 
    w6 = zeros(member, p) 
    w7 = zeros(member, member) 
    ########################################## 
    # localization matrix 
    #xi, yi, zi, xj, yj, zj = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 
    #d = 0.0 
    #println("----- calc localization matrix -------") 
    #@time rho(rho_mat, D, nCells, N, d, xi, yi, zi, xj, yj, zj) 
    #println("----- finish calc localization matrix -------") 
    println("==== read rho matrix =========") 
    file = h5open("./rhoMatrix.h5","r")  
    rho_mat = read(file, "rho")  
     
    ####################################### 


    # make H matrix, y vector 
    makeH(H, obsFlag, nCells, nObs, p) 
    makey(y, ftime) 
     
    # make ensemble member vector 
    makeEnsemble(Xf, nCells, member, ftime, U1, U2, U3, T) 
     
    # member average 
    mean!(w4, Xf) 
    for i=1:N 
        Xf_m[i] = w4[i] 
    end 
    #println(Xf) 
     
    # calc dXf 
    for l=1:member 
        for i=1:N 
            dXf[i,l] = Xf[i,l] - Xf_m[i] 
        end 
    end 
     
    # inflation 
    @. dXf = alpha*dXf 
    println("------- finish inflation-------------") 
    # dY 
    mul!(dY, H, dXf)  
    Pf .= mul!(w3, dXf, dXf')./ (member-1.0) 
    # localization 
    Pf .= rho_mat .* Pf 
    println("------- finish localization -------------") 


    # K=dXf*dY'*inv(dY*dY' + (m-1)R) 
    #mul!(w1, dXf, dY')  
    #w2_1 .= mul!(w2, dY, dY') .+ (member-1).*R             
    #w2_1 .= inv(w2_1) # w2_1 = inv(dY*dY' + (m-1)R) 
    #mul!(K, w1, w2_1) 


    # K = (m-1)*Pf*H'*inv((m-1)*H*Pf*H' + (m-1)*R) 
    mul!(w1, Pf, H')  
    w2_1 .= (member-1).*mul!(w2, H, w1) .+ (member-1).*R             
    w2_1 .= inv(w2_1) #w2_1 = inv((m-1)H*Pf*H' + (m-1)R) 
    mul!(K, w1, w2_1) 
    K .= (member-1).*K 


    # Xa_m = Xf_m + K(y-H*Xf_m) 
    mul!(w5, H, Xf_m) 
    w5 .= y .- w5 
    mul!(dX, K, w5) 
    for i=1:N 
        Xa_m[i] = Xf_m[i] + dX[i] 
    end 
     
    # val, vec = eigen(I - dY'*inv(dY*dY'+(m-1)*R)*dY) 
    mul!(w2, dY, dY') 
    w2_1 .= w2 .+ (member-1).* R 
    w2_1 .= inv(w2_1) 
    mul!(w6, dY', w2_1) 
    mul!(w7, w6, dY) 
    w7 .= E .- w7 
             
    val, vec = eigen(Symmetric(w7)) 
             
    mul!(w7, vec, sqrt(Diagonal(val))) 
    mul!(Tr, w7, vec') 
    mul!(dXa, dXf, Tr) 
             
    for l=1:member 
        for i=1:N 
            Xa[i,l] = Xa_m[i] + dXa[i,l] 
        end 
    end 
    #println(Xa) 


    # output Xa data to "dataassim.tx" 
    writeXa(Xa, member, ftime) 


    writeXam(Xa_m, ftime) 
    writedXa(dXa, Pf, N, ftime) 
end 




############################################ 
function makeEnsemble(Xf, nCells, member, ftime, U1, U2, U3, T) 
    for i=1:member 
        dir = "dataassim/dataassim_"*string(i-1)*"/"*ftime 
        read_vector(dir*"/U", nCells, U1, U2, U3) 
        read_scalar(dir*"/T", nCells, T) 
        for j=1:nCells 
            Xf[j, i] = U1[j] 
            Xf[j+nCells, i] = U2[j] 
            Xf[j+2nCells, i] = U3[j] 
            Xf[j+3nCells, i] = T[j] 
        end 
    end 
end 


function makeH(H, obsFlag, nCells, nObs, p) 
    numObsPos= Int(p/nObs) # number of observed positions 
    listFlag = findall(x->(x==1), obsFlag) # elment number of observed quatinties 
     
    poso = CSV.read("pos_obs.csv", DataFrame, header=false) 
    row= 1 
     
    for i in listFlag         
        for point in eachrow(poso[row:row+numObsPos-1,:]) 
            x1, y1, z1 = point["Column1"], point["Column2"], point["Column3"] 
         
            d = 1e5 
            columns = (i-1)*nCells + 1 
            column = 0 
            counter = 0 
         
            for cellp in eachline("cellCoord.csv")          
                cpositions = parse.(Float64,split(cellp, ",")) 
                x2, y2, z2 = cpositions[1], cpositions[2], cpositions[3] 
             
                d_new = distance2(x1,y1,z1,x2,y2,z2) 
                #println(x1, y1, ", ", x2, y2) 
                #println(d, d_new) 
                if d_new <= d 
                    d = d_new 
                    column = counter 
                end 
                counter = counter + 1         
            end 
            column = column + columns 
            H[row, column] = 1.0 
            #println("r, c=", obsrow,", ", column) 
             
            row = row + 1     
        end    
    end 
end 


function distance2(x1,y1,z1,x2,y2,z2) 
    return (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 
end 


function makey(y, ftime) 
    yall = CSV.read("obs_data.csv", DataFrame) 
    y .= yall[!, ftime] 
end 


function read_scalar(file, ncells, T) 
    str = open( file, "r" ) do fp 
        read( fp, String ) 
    end 
    data = split(split(str, "(\n")[2], "\n)")[1] 
    T .= parse.(Float64, split(data, "\n")) 
end 


function read_vector(file, ncells, U1, U2, U3) 
    strU = open( file, "r" ) do fp 
        read( fp, String ) 
    end 
    uarry = split(split(strU, "(\n")[2], "\n)")[1] 
    U=split(uarry, "\n") 
     
    for i=1:ncells 
        u = parse.(Float64, split(split(split(U[i], "(")[2], ")")[1], " ")) 
        U1[i], U2[i], U3[i] = u[1], u[2], u[3] 
    end 
end 


# output 
function writeXa(Xa, member, ftime) 
    for i=1:member 
        dir = "dataassim/dataassim_"*string(i-1)*"/"*ftime 
        open(dir*"/dataassim.txt","w") do out 
            Base.print_array(out, Xa[:,i])  
        end 
    end 
end 


function writeXam(Xa_m, ftime) 
    dir = "dataassim/"*ftime 
    mkdir(dir) 
    open(dir*"/analysis.txt","w") do out 
        Base.print_array(out, Xa_m)  
    end 
end 




function writedXa(dXa, Pf, N, ftime) 
    dir = "dataassim/"*ftime 
    open(dir*"/dXa.txt","w") do out 
        Base.print_array(out, dXa)  
    end 
     
    dir = "dataassim/"*ftime 


    h5open(dir*"/Pf.h5","w") do out 
        write(out, "Pf", Pf[1:50:N, 1:50:N] )  
        #write(out, "Pf_temp", Pf[Int(3*N//4)+1:N, Int(3*N//4)+1:N] )  
    end 


    open("dataassim/trPf.txt","a") do out 
        println(out, ftime, " ", sqrt(tr(Pf)/N) )  
    end 
end 

