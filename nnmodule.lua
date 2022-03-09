local Matrix = {}

function random(min, max)
	min = min or 0
	max = max or 1
	return min + math.random() * (max - min)
end

function module.new(rows, cols)
	-- Creates a new Matrix with x rows and y coloums
	local Table = {}
	Table.rows = rows
	Table.cols = cols
	Table.Matrix = {}
	for x = 1,rows do
		Table.Matrix[x] = {}
		for y = 1,cols do
			Table.Matrix[x][y] = 0
		end
	end
	return Table
end

function module:addMatrices(M1, M2)
	local NewMatrix = module.new(M1.rows, M1.cols)
	for x = 1,NewMatrix.rows do
		for y = 1,NewMatrix.cols do
			NewMatrix.Matrix[x][y] = M1.Matrix[x][y] + M2.Matrix[x][y]
		end
	end
	return NewMatrix
end

function module:randomizeMatrix(M, min, max)
	for x = 1,M.rows do
		for y = 1,M.cols do
			M.Matrix[x][y] = random(min, max)
		end
	end
end

function module:subMatrices(M1, M2)
	local NewMatrix = module.new(M1.rows, M1.cols)
	for x = 1,NewMatrix.rows do
		for y = 1,NewMatrix.cols do
			NewMatrix.Matrix[x][y] = M1.Matrix[x][y] - M2.Matrix[x][y]
		end
	end
	return NewMatrix
end

function module:toMatrix(array)
	local e = array["Matrix"]
	if type(e) == "table" then
		return array
	elseif e == nil then
		local arrayToMatrix = module.new(#array, 1)
		for x = 1,#array do
			arrayToMatrix.Matrix[x][1] = array[x]
		end
		
		return arrayToMatrix
	end
end

function module:toArray(M)
	local array = {}
	for x = 1,M.rows do
		for y = 1,M.cols do
			array[#array+1] = M.Matrix[x][y]
		end
	end
	return array
end

function module:divMatrices(M1, M2)
	local NewMatrix = module.new(M1.rows, M1.cols)
	for x = 1,NewMatrix.rows do
		for y = 1,NewMatrix.cols do
			NewMatrix.Matrix[x][y] = M1.Matrix[x][y] / M2.Matrix[x][y]
		end
	end
	return NewMatrix
end

function module:transposeMatrix(M)
	local newMatrix = module.new(M.cols, M.rows)
	for x = 1,M.rows do
		for y = 1,M.cols do
			newMatrix.Matrix[y][x] = M.Matrix[x][y]
		end
	end
	return newMatrix
end

function module:multMatrices(M1, M2, scalar)
	if type(M2) == "number" and type(M1) == "table" then
		local newMatrix = module.new(M1.rows, M1.cols)
		for x = 1,newMatrix.rows do
			for y = 1,newMatrix.cols do
				newMatrix.Matrix[x][y] = M1.Matrix[x][y] * M2
			end
		end
		return newMatrix
	elseif type(M1) == "number" and type(M2) == "table" then
		local newMatrix = module.new(M2.rows, M2.cols)
		for x = 1,newMatrix.rows do
			for y = 1,newMatrix.cols do
				newMatrix.Matrix[x][y] = M2.Matrix[x][y] * M1
			end
		end
		return newMatrix
	end
	
	if type(M1) == "table" and type(M2) == "table" and scalar == true then
		local newMatrix = module.new(M1.rows, M1.cols)
		for x = 1,newMatrix.rows do
			for y = 1,newMatrix.cols do
				newMatrix.Matrix[x][y] = M1.Matrix[x][y] * M2.Matrix[x][y]
			end
		end
		return newMatrix
	elseif type(M1) == "table" and type(M2) == "table" and not scalar then
		local newMatrix = module.new(M1.rows, M2.cols)
		for x = 1,newMatrix.rows do
			for y = 1,newMatrix.cols do
				local sum = 0
				for z = 1,M1.cols do
					sum += M1.Matrix[x][z] * M2.Matrix[z][y]
				end
				newMatrix.Matrix[x][y] = sum
			end
		end
		return newMatrix
	end
end

function module:mapMatrix(M, functio, s)
	local newMatrix = module.new(M.rows, M.cols)
	for x = 1,newMatrix.rows do
		for y = 1,newMatrix.cols do
			newMatrix.Matrix[x][y] = functio(M.Matrix[x][y], s)
		end
	end
	return newMatrix
end




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

ActivationFuncs.LeakyReLU = function(x, d)
	if d then
		if x >= 0 then
			return 1
		end
		return 0.1
	end
	return math.max(0.1*x, x)
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
	
	if a == "LeakyReLU" then
		if d then
			if x >= 0 then
				return 1
			end
			return 0.1
		end
		return math.max(0.1*x, x)
	end
end

function module:CopyTable(s)
	local f = {}
	for i = 1,#s do
		f[i] = s[i]
	end
	return f
end

function module:LayerM(net, input, layer, f, d, s)
	local minput = Matrix:toMatrix(input)
	if f then
		local n = Matrix:multMatrices(net.__Weights[layer], minput)
		local nplusb = Matrix:addMatrices(n, net.__Bias[layer])
		local nplusb_a = Matrix:mapMatrix(nplusb, ActivationFuncs[net.__AF], d)
		return nplusb_a
	else
		if s then
			return Matrix:multMatrices(Matrix:transposeMatrix(net.__Weights[layer-1]), minput)
		end
		local n = Matrix:multMatrices(Matrix:transposeMatrix(net.__Weights[layer-1]), minput)
		local nplusb = Matrix:addMatrices(n, net.__Bias[layer-1])
		local nplusb_a = Matrix:mapMatrix(nplusb, ActivationFuncs[net.__AF], d)
		return nplusb_a
	end
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
	NewNN.__Score = 0
	NewNN.__Hidden = {}
	
	NewNN.__Layers[1] = NumberInputs
	
	for layer = 1,NumberHLayers,1 do
		NewNN.__Bias[layer] = Matrix.new(NumberHidden, 1)
		NewNN.__Layers[#NewNN.__Layers+1] = NumberHidden
	end

	NewNN.__Bias[NumberHLayers+1] = Matrix.new(NumberOutputs, 1)
	
	NewNN.__Layers[#NewNN.__Layers+1] = NumberOutputs
	
	local iw = Matrix.new(NumberHidden, NumberInputs)
	Matrix:randomizeMatrix(iw, -1, 1)
	NewNN.__Weights[1] = iw
	
	for i=1,NumberHLayers-1 do
		local hw = Matrix.new(NumberHidden, NumberHidden)
		Matrix:randomizeMatrix(hw, -1, 1)
		NewNN.__Weights[#NewNN.__Weights+1] = hw
	end
	
	local ow = Matrix.new(NumberOutputs, NumberHidden)
	Matrix:randomizeMatrix(ow, -1, 1)
	
	NewNN.__Weights[#NewNN.__Weights+1] = ow
	
	function NewNN:Forward(Inputs, giveotherstuff, d)
		local hidden = {}
		local outp = {}
		local output = {}
		local otherstuff = {}
		otherstuff[1] = Inputs
		local oh = module:LayerM(self, Inputs, 1, true)
		if giveotherstuff then
			otherstuff[#otherstuff+1] = Matrix:toArray(oh)
		end
		local inpu = oh
		
		for l = 1,self.__NumberHLayers,1 do
			inpu = module:LayerM(self, inpu, l+1, true)
			if giveotherstuff then
				otherstuff[#otherstuff+1] = Matrix:toArray(inpu)
			end
		end
		
		output = inpu
	
		
		if giveotherstuff then
			return Matrix:toArray(output), otherstuff
		end
		return Matrix:toArray(output)
	end
	
	function NewNN:BackProp(Inputs, Targets)
		local mTargets = Matrix:toMatrix(Targets)
		local Output, stuff = self:Forward(Inputs, true)
		local mOutput = Matrix:toMatrix(Output)
		local l_error = Matrix:subMatrices(mTargets, mOutput)
		for layer = self.__NumberHLayers+2, 2, -1 do
			local vinlayer = Matrix:toMatrix(stuff[layer])
			local lvinlayer = Matrix:toMatrix(stuff[layer-1])
			local t_lvinlayer = Matrix:transposeMatrix(lvinlayer)
			local g = Matrix:mapMatrix(vinlayer, ActivationFuncs[self.__AF], true)
			g = Matrix:multMatrices(g, l_error, true)
			g = Matrix:multMatrices(g, self.__LR)
			self.__Weights[layer-1] = Matrix:addMatrices(self.__Weights[layer-1], Matrix:multMatrices(g, t_lvinlayer))
			self.__Bias[layer-1] = Matrix:addMatrices(self.__Bias[layer-1], g)
			l_error = module:LayerM(self, l_error, layer, false, false, true)
		end
		
	end
	
	function NewNN:RandomizeWeights()
		for i, v in pairs(NewNN.__Weights) do
			for ii, vv in pairs(v) do
				self.__Weights[i][ii] = random(-1, 1)
			end
		end
	end
	return NewNN
end

function module:CopyNet(net)
	local copiednet = module:CreateNN(net.__NumberInputs, net.__NumberHidden, net.__NumberHLayers, net.__NumberOut, net.__LR, net.__AF)
	return copiednet
end

function module:MutateNet(net, c, chance, rate)
	chance = chance or 50
	for i, v in pairs(net.__Weights) do
		for ii, vv in pairs(v) do
			local random = (math.random()*2-1)*rate
			if c then
				if random > (chance/100-.5) then
					self.__Weights[i][ii] += random
				end
			else
				self.__Weights[i][ii] += random
			end
		end
	end
	
	for i, v in pairs(net.__Bias) do
		for ii, vv in pairs(v) do
			local random = (math.random()*2-1)*rate
			if c then
				if random > (chance/100-.5) then
					self.__Bias[i][ii] += random
				end
			else
				self.__Bias[i][ii] += random
			end
		end
	end
end

function module:CrossNets(net1, net2, m, rate)
	local child = module:CopyNet(net1)
	for i, v in pairs(net1.__Weights) do
		for ii, vv in pairs(v) do
			local p1w = net1.__Weights[i][ii]
			local p2w = net2.__Weights[i][ii]
			if m then
				local random = random(-1, 1)*(rate or 1)
				child.__Weights[i][ii] = ((p1w + p2w)/2) + random
			else
				child.__Weights[i][ii] = (p1w + p2w)/2
			end
		end
	end
	
	for i, v in pairs(net1.__Bias) do
		for ii, vv in pairs(v) do
			local p1w = net1.__Bias[i][ii]
			local p2w = net2.__Bias[i][ii]
			if m then
				local random = random(-1, 1)*(rate or 1)
				child.__Bias[i][ii] = ((p1w + p2w)/2) + random
			else
				child.__Bias[i][ii] = (p1w + p2w)/2
			end
		end
	end
	return child
end

function module:CreateGA(nettopology, batchsize)
	local nets = {}
	for i = 1,batchsize do
		local copied = module:CopyNet(nettopology)
		copied:RandomizeWeights()
		nets[i] = copied
	end
	return nets
end

function module:RunGAFitness(GA, scores)
	for i = 1,#GA do
		GA[i].__Score = scores[i]
	end
end

function module:RunGA(GA, scores)
	module:RunGAFitness(GA, scores)
	local netscores = {}
	local fittestnet
	local children = {}
	local child
	netscores = module:CopyTable(scores)
	table.sort(scores)
	fittestnet = table.find(netscores, scores[#scores])
	for i = 1,#GA do
		children[#children+1] = module:MutateNet(GA[fittestnet], false, 0, .1)
	end
	GA = module:CopyTable(children)
	return fittestnet
end

return module

