require 'torch'
require 'nn'
require 'nnx'
require 'cunn'

print '=> 2_model.lua'
print '<2_model.lua>: Defining CNN model'
-- features size
fSize = {1, 96, 256, 256, 256}
featuresOut = fSize[5] * 3 * 3

-- classifier size
classifierHidden = {512}
dropout_prob = 0.5

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(fSize[1], fSize[2], 7, 7, 2, 2)) -- (110-7+2)/2=52
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 26

features:add(nn.SpatialConvolutionMM(fSize[2], fSize[3], 5, 5)) -- 22
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 11

features:add(nn.SpatialConvolutionMM(fSize[3], fSize[4], 4, 4)) -- 8
features:add(nn.Threshold(0,1e-6))

features:add(nn.SpatialConvolutionMM(fSize[4], fSize[5], 3, 3)) -- 6
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 3
features:add(nn.View(featuresOut))-- features size

full_layer = nn.Sequential()

if opt.dropout then
   dropouts = nn.Dropout(dropout_prob)
   full_layer:add(dropouts)
end
full_layer:add(nn.Linear(featuresOut, classifierHidden[1]))
full_layer:add(nn.Threshold(0, 1e-6))

if opt.dropout then
   dropouts = nn.Dropout(dropout_prob)
   full_layer:add(dropouts)
end
full_layer:add(nn.Linear(classifierHidden[1], nClasses))
full_layer:add(nn.LogSoftMax())

model = nn.Sequential()
model:add(features)
model:add(full_layer)

criterion = nn.ClassNLLCriterion()
