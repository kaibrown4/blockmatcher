local module = {}

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

return module
