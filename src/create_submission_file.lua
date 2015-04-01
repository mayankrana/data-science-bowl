require 'csvigo'
require 'optim'
require 'paths'
require 'nn'
require 'cunn'
require 'image'
require 'xlua'

dofile('1_datafunctions.lua')

dataRoot = "../data/"
submission_file = "./results/exp_1_9_2/submission_file.csv"
model_path = "./results/exp_1_9_2/model_80.net"

sampleSize = {1, 48, 48}

local i=1

function getClassInfo()
   print "Getting class info"
   local n_classes = 0
   local class_id_to_name = {}
   local class_name_to_id = {}
   if paths.filep(dataRoot .. 'cache/class_info.dat') then
      print('<1_data.lua>: Cached class info found, Loading from cache')
      local class_info = torch.load(dataRoot .. 'cache/class_info.dat')
      class_name_to_id = class_info["class_name_to_id"]
      class_id_to_name = class_info["class_id_to_name"]  
      n_classes = class_info["n_classes"]
   end
   return class_id_to_name, class_name_to_id, n_classes
end

local function add_header_to_table(n_classes, class_id_to_name)
   print "Adding Header"
   local output_table = {}
   output_table[1] = {}
   output_table[1][1] = "image"
   for i=1,n_classes do
       output_table[1][i+1] = class_id_to_name[i]
   end
   return output_table
end

function testAndCreateOutput(n_classes, class_id_to_name)
   local output_table = add_header_to_table(n_classes, class_id_to_name)

   print "Loading Model"
   --Load the model
   model = torch.load(model_path):float()
   model:cuda()

   --Get name of all the images and iteratively read them
   local test_dir_name = paths.concat(dataRoot, 'test')
   local file_name
   local idx = 1
   for file_name in paths.files(test_dir_name) do
     xlua.progress(idx, 140000)

      if string.sub(file_name, -4) == ".jpg" then
      	 local file_path = paths.concat(test_dir_name,file_name)
	 --print('Reading file ' .. file_path)
	 local _im = image.load(file_path, 1)
	 _im = random_square_crop(_im)
	 _im = scaleToSampleSize(_im)
	 _im = dataNormalization(_im)
	 _im = _im:cuda()

   	--Run the model for this image
	local _output = model:forward(_im)
	_output = _output:float()

	--convert log prob to probs
	_output = torch.exp(_output)
	
	--Store the prob value for each class
	output_table[idx+1] = {}
	output_table[idx+1][1] = file_name
   	for i=1,n_classes do
       	    output_table[idx+1][i+1] = _output[i]
	end
	idx = idx + 1
      end
      if(math.fmod(idx,5000)==0) then
  	collectgarbage()
      end
   end
   return output_table
end

function saveResultsToCsv(table)
   print "Saving submission file"
   csvigo.save{path=submission_file,
		data=table,
		mode='raw',
		header=true,
		verbose=false}
end

function test()
   local class_id_to_name, class_name_to_id, n_classes = getClassInfo()
   local output_table = testAndCreateOutput(n_classes, class_id_to_name)
   saveResultsToCsv(output_table)
end


test()

