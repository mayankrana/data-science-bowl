require 'csvigo'
require 'paths'
require 'image'
dofile('1_datafunctions.lua')

dataRoot='../data/'

os.execute('mkdir -p ../data/cache')
print '=> 1_data.lua'
print('-----------')
print '==> <1_data.lua>: Loading data'

nSamples=0
nClasses=0
local i=1

function getClassInfo()
   local class_id_to_name = {}
   local class_name_to_id = {}
   local n_classes = 0
   if not paths.filep(dataRoot .. 'cache/class_info.dat') then
      --Read the sample submission csv to get the names of all the classes and
      -- assign an id to classes
      local csv_data = csvigo.load{path=dataRoot .. 'sampleSubmission.csv',
                                   mode='raw',
                                   verbose=false}
      -- first entity is "image"(not a class name) so skip it
      n_classes = #csv_data[1] - 1
      print('<1_data.lua>: Number of classes=' .. n_classes)
      for i=1,n_classes do
         class_id_to_name[i] = csv_data[1][i+1]
         class_name_to_id[csv_data[1][i+1]] = i
      end
      local class_info = {["class_id_to_name"]=class_id_to_name,
         ["class_name_to_id"]=class_name_to_id,
         ["n_classes"]=n_classes}
      print '<1_data.lua>: Saving class info to disk'
      torch.save(dataRoot .. 'cache/class_info.dat',class_info)
   else
      print('<1_data.lua>: Cached class info found, Loading from cache')
      local class_info = torch.load(dataRoot .. 'cache/class_info.dat')
      class_name_to_id = class_info["class_name_to_id"]
      class_id_to_name = class_info["class_id_to_name"]
      n_classes = class_info["n_classes"]
   end
   return class_id_to_name, class_name_to_id, n_classes
end

classIdToName, classNameToId, nClasses = getClassInfo()

--If data(tensor of image_name to class_id) is not already created
-- then create and save it
if not paths.filep(dataRoot .. 'cache/data.t7') then
   --manually setting it, need it for tensor initialization beforehand
   --TODO: This is bad, figure out programatically
   nSamples=30336
   data=torch.Tensor(nSamples,2):zero()
   i=1
   for key, value in pairs(classIdToName) do
      local class_dir_name = paths.concat(dataRoot, 'train', value)
      for file_name in paths.files(class_dir_name) do
         --replace extension ".jpg" with file id
         if string.sub(file_name, -4) == ".jpg" then
            file_name = string.sub(file_name, 1, string.len(file_name) - 4)
            data[i][1] = tonumber(file_name)
            data[i][2] = key
            i = i+1
         end
      end
   end
   print '<1_data.lua>: Saving data to disk'
   torch.save(dataRoot .. 'cache/data.t7', data)
else
   print('<1_data.lua>: Cached data found, Loading from cache')
   data = torch.load(dataRoot .. 'cache/data.t7')
   nSamples = data:size(1)
end

key = nil
value = nil
file_names = nil

-- split into training/testing 90/10
nTraining = math.floor(nSamples * 0.90)
nTesting = nSamples - nTraining

local randIndices = torch.randperm(nSamples)
local trainingIndices = randIndices[{{1,nTraining}}]
local testIndices = randIndices[{{nTraining+1,nSamples}}]

trainData = torch.Tensor(nTraining, 2):zero()
testData = torch.Tensor(nTesting, 2):zero()

for i=1,nTraining do
   trainData[i] = data[trainingIndices[i]]
end

for i=1,nTesting do
   testData[i] = data[testIndices[i]]
end

i=nil
collectgarbage()
--=======================
print('<1_data.lua>: Number of Samples: ' .. nSamples)
print('<1_data.lua>: Training samples: ' .. nTraining)
print('<1_data.lua>: Testing samples: ' .. nTesting)

--TODO Check the random pick from pool if added back to pool
function getSample()
   local _i = torch.uniform(1, nTraining)
   local _filename = paths.concat(dataRoot,
                                 'train',
                                 classIdToName[trainData[_i][2]],
                                 tostring(trainData[_i][1]) .. '.jpg')
   local _im = image.load(_filename, 1)
   _im = dataAugmentation(_im)
   _im = scale(_im)
   _im = dataNormalization(_im)
   return _im, trainData[_i][2]
end

function getBatch(n)
   local _img, _labels
   _img = torch.Tensor(n, sampleSize[1],sampleSize[2], sampleSize[3])
   _labels = torch.Tensor(n)
   for i=1,n do
      _img[i], _labels[i] = getSample()
   end
   return _img, _labels
end

function getTest(_i, light_testing)
   local _filename = paths.concat(dataRoot,
                                 'train',
                                 classIdToName[testData[_i][2]],
                                 tostring(testData[_i][1]) .. '.jpg')
   local _im = image.load(_filename, 1)
--   im = expandTestSample(im, lightTesting)
   _im=dataAugmentation(_im)
   _im = random_crop(_im)
   _im = scale(_im)
   _im=dataNormalization(_im)

   return _im, testData[_i][2]
end
