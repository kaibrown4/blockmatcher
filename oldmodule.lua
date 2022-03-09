local module = {}
 
function random(min, max)
    return min + math.random() * (max-min)
end
 
local ActivationFuncs = {}
 
ActivationFuncs.Sigmoid = function(x, d)
    if d then
        return x*(1-x)
    end
    return 1/(1+math.exp(-x))
end
 
ActivationFuncs.Tanh = function(x, d)
    if d then
        return 1/math.cosh(x)^2
    end
    return math.tanh(x)
end
 
ActivationFuncs.Identity = function(x, d)
    if d then
        return 1
    end
    return x
end
 
ActivationFuncs.ArcTanh = function(x, d)
    if d then
        return 1/(x^2+1)
    end
    return math.atan(x)
end
 
ActivationFuncs.Sign = function(x, d)
    if d then
        return 0
    end
    return math.sign(x)
end
 
function module:MAp(tabl, functio, d)
    for i, v in pairs(tabl) do
        if type(v) == "table" then
            module:MAp(tabl[i], functio)
        elseif type(v) == "number" then
            tabl[i] = functio(tabl[i], d)
        end
    end
end
 
function module:ActivationFunc(x, d, a)
    if a == "Sigmoid" then
        if d then
            return x*(1-x)
        end
        return 1/(1+math.exp(-x))
    end
 
    if a == "Identity" then
        if d then
            return 1
        end
        return x
    end
 
    if a == "Tanh" then
        if d then
            return 1/math.cosh(x)^2
        end
        return math.tanh(x)
    end
 
    if a == "ATanh" then
        return (math.tanh(x)+1)/2
    end
 
    if a == "ArcTanh" then
        if d then
            return 1/(x^2+1)
        end
        return math.atan(x)
    end
 
    if a == "MO" then
        if d then
            return 0
        end
        if x > 0 then
            return 1
        end
        return 0
    end
 
    if a == "Sign" then
        return math.sign(x)
    end
end
 
function module:LayerM(net, input, layer, f, d)
    local out = {}
    if f then
        for l = 1,net.__Layers[layer+1] do
            local sum = 0
 
            for inpu = 1,#input do
                local inp = input[inpu]
                local weig = net.__Weights[layer][inpu*l]
                sum += inp * weig
            end
            
            sum += net.__Bias[layer][l]
            
            table.insert(out, module:ActivationFunc(sum, d, net.__AF))
        end
    else
        for l = 1,net.__Layers[layer-1] do
            local sum = 0
 
            for inpu = 1,#input do
                local inp = input[inpu]
                local weig = net.__Weights[layer-1][inpu*l]
                sum += inp * weig
            end
            
            sum += (net.__Bias[layer-1][l] or 0)
            table.insert(out, module:ActivationFunc(sum, d, net.__AF))
        end 
    end
    return out
end
 
function module:CreateNN(NumberInputs, NumberHidden, NumberHLayers, NumberOutputs, LearningRate, ActivationFunction)
    local NewNN = {}
    
    NewNN.__NumberInputs = NumberInputs
    NewNN.__NumberHidden = NumberHidden
    NewNN.__NumberHLayers = NumberHLayers
    NewNN.__NumberOut = NumberOutputs
    NewNN.__LR = LearningRate
    NewNN.__AF = ActivationFunction
    NewNN.__Weights = {}
    NewNN.__Layers = {}
    NewNN.__Bias = {}
    NewNN.__Hidden = {}
    NewNN.__Weights[1] = {}
    
    NewNN.__Layers[1] = NumberInputs
    
    for layer = 1,NumberHLayers,1 do
        NewNN.__Bias[layer] = {}
        NewNN.__Layers[#NewNN.__Layers+1] = NumberHidden
        for node = 1,NumberHidden,1 do
            NewNN.__Bias[layer][node] = 0
        end
    end
    NewNN.__Bias[NumberHLayers+1] = {}
    for out = 1,NumberOutputs,1 do
        NewNN.__Bias[NumberHLayers+1][out] = random(-1,1)
    end
    
    NewNN.__Layers[#NewNN.__Layers+1] = NumberOutputs
    
    for i = 1,NumberInputs*NumberHidden do
        NewNN.__Weights[1][i] = random(-1, 1)
    end
    
    for i=1,NumberHLayers-1 do
        NewNN.__Weights[i+1] = {}
        for ii = 1,NumberHidden^2 do
            NewNN.__Weights[i+1][ii] = random(-1, 1)
        end
    end
    
    NewNN.__Weights[#NewNN.__Weights+1] = {}
    
    for i=1,NumberHidden*NumberOutputs do
        NewNN.__Weights[#NewNN.__Weights][i] = random(-1, 1)
    end
    
    function NewNN:Forward(Inputs, giveotherstuff, d)
        local hidden = {}
        local outp = {}
        local output = {}
        local otherstuff = {}
        otherstuff[1] = Inputs
        local oh = module:LayerM(self, Inputs, 1, true)
        if giveotherstuff then
            otherstuff[#otherstuff+1] = oh
        end
        local inpu = oh
        
        for l = 1,self.__NumberHLayers,1 do
            inpu = module:LayerM(self, inpu, l+1, true)
            if giveotherstuff then
                otherstuff[#otherstuff+1] = inpu
            end
        end
        
        output = inpu
    
        
        if giveotherstuff then
            return output, otherstuff
        end
        return output
    end
    
    function NewNN:BackProp(Inputs, Targets)
        local Output, stuff = self:Forward(Inputs, true)
        local out_errors = {}
        local DerivativeOut = {}
        local OME = {}
        local HES = {}
        local OUTLR = {}
        local DeltaWeights = {}
        local e = {}
        local tt = 0
        for OutputNode = 1,self.__NumberOut do
            DerivativeOut[OutputNode] = module:ActivationFunc(Output[OutputNode], true, self.__AF)
            out_errors[OutputNode] = Targets[OutputNode] - Output[OutputNode]
            OME[OutputNode] = out_errors[OutputNode] * DerivativeOut[OutputNode]
            OUTLR[OutputNode] = OME[OutputNode] * self.__LR
        end
        local HE = module:LayerM(self, out_errors, self.__NumberHLayers+2, false, true)
        local E = module:LayerM(self, out_errors, self.__NumberHLayers+2)
        local L = HE
        HES[1] = HE
        for Node = 1,#stuff[#stuff-1] do
            for out = 1,#OUTLR do
                e[#e+1] = stuff[#stuff-1][Node] * OUTLR[out]
            end
        end
        for Layer = self.__NumberHLayers+1,2,-1 do
            HES[#HES+1] = E
            for dl = 1,#L do
                for node = 1,#stuff[Layer-1] do
                    local stuffinlayer = stuff[Layer]
                    module:MAp(stuffinlayer, ActivationFuncs[self.__AF], true)
                    e[#e+1] = E[dl] * stuffinlayer[dl] * stuff[Layer-1][node] * self.__LR
                end
            end
            L = module:LayerM(self, L, Layer, false, true)
            
            E = module:LayerM(self, E, Layer, false)
        end
        DeltaWeights = e
        for o = 1,self.__Layers[#self.__Layers] do
            local db = self.__Bias[#self.__Bias][o] + OUTLR[o]
            if db > 1e10 or db < -1e10 then
                self.__Bias[#self.__Bias][o] = random(-2, 2)
            else
                self.__Bias[#self.__Bias][o] += OUTLR[o]
            end
        end
        for la = 1,self.__NumberHLayers do
            for no = 1,self.__NumberHidden do
                local db = self.__Bias[la][no] + HES[la][no] * self.__LR
                if db > 1e10 or db < -1e10 then
                    self.__Bias[la][no] = random(-2, 2)
                else
                    self.__Bias[la][no] += HES[la][no] * self.__LR
                end
            end
        end
        for i, v in pairs(self.__Weights) do
            for ii, vv in pairs(v) do
                local Weight = self.__Weights[i][ii]
                tt+=1
                if Weight+DeltaWeights[tt] > 1e10 or Weight+DeltaWeights[tt] < -1e30 then
                    self.__Weights[i][ii] = random(-1, 1)
                else
                    self.__Weights[i][ii] += DeltaWeights[tt]
                end
            end
        end
    end
    
    return NewNN
end

return module
