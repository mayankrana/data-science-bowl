require 'nn'
require 'nnx'

print '=> 2_model.lua'
print '==>  Defining CNN model'
-- features size
fSize = {1, 96, 256, 256, 256}
featuresOut = fSize[5] * 3 * 3

-- classifier size
classifierHidden = {512}
dropout_prob = 0.5

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(fSize[1], fSize[2], 7, 7)) -- (90 - 7 + 1)=84
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 42
features:add(nn.SpatialConvolutionMM(fSize[2], fSize[3], 5, 5)) -- 38
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 19
features:add(nn.SpatialConvolutionMM(fSize[3], fSize[4], 4, 4)) -- 16
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 8
features:add(nn.SpatialConvolutionMM(fSize[4], fSize[5], 3, 3)) -- 6
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 3
features:add(nn.View(featuresOut))-- features size

classifier_layer = nn.Sequential()
classifier_layer:add(nn.Linear(featuresOut, classifierHidden[1]))
classifier_layer:add(nn.Threshold(0, 1e-6))
classifier_layer:add(nn.Linear(classifierHidden[1], nClasses))
classifier_layer:add(nn.LogSoftMax())

model = nn.Sequential()
model:add(features)
model:add(classifier_layer)

criterion = nn.ClassNLLCriterion()
