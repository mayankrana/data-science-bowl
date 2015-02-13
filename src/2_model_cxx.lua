require 'torch'
require 'nn'
require 'nnx'
require 'cunn'

print '=> 2_model.lua'
print '<2_model.lua>: Defining CNN model'
-- features size
fSize = {1, 96, 128, 128}
featuresOut = fSize[5] * 3 * 3

-- classifier size
classifierHidden = {512,512}
dropout_prob = 0.5

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(fSize[1], fSize[2], 5, 5, 4, 4, 2)--11
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(3,3,2,2)) -- 5
---------------
features:add(nn.SpatialConvolutionMM(fSize[2], fSize[3], 3, 3, 1, 1, 2)) -- 7
features:add(nn.Threshold(0,1e-6))
---------------
features:add(nn.SpatialConvolutionMM(fSize[3], fSize[4], 3, 3, 1, 1, 1)) -- 7
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(3,3,2,2)) -- 3
------
features:add(nn.View(featuresOut))-- features size


full_layer = nn.Sequential()

full_layer:add(nn.Linear(featuresOut, classifierHidden[1]))
full_layer:add(nn.Threshold(0,1e-6))

if opt.dropout then
   dropouts = nn.Dropout(dropout_prob)
   full_layer:add(dropouts)
end

full_layer:add(nn.Linear(classifierHidden[1], classifierHidden[2]))
full_layer:add(nn.Threshold(0, 1e-6))

if opt.dropout then
   dropouts = nn.Dropout(dropout_prob)
   full_layer:add(dropouts)
end

full_layer:add(nn.Linear(classifierHidden[2], nClasses))
full_layer:add(nn.LogSoftMax())

model = nn.Sequential()
model:add(features)
model:add(full_layer)

criterion = nn.ClassNLLCriterion()
