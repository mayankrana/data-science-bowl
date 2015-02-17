require 'torch'
require 'xlua'
require 'optim'
require 'image'

print '=> 4_test.lua'
print '<4_test.lua>: Defining test procedure'
testRMSELogger = optim.Logger(paths.concat(opt.results_path, 'testRMSE.log'))
testNLLLogger = optim.Logger(paths.concat(opt.results_path, 'testNLL.log'))

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(class_id_to_name)

-- test function
function test()
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   -- test over test data
   print('<4_test.lua>: Testing on test set:')
   local _mse = 0
   local nll_error = 0
   for _t = 1,nTesting do
      if opt.progressBar then xlua.progress(_t, nTesting) end
      -- test sample
      -- target is sparse label
      local _input, _target, _label = getTest(_t, lightTesting)
      _input = _input:cuda()
      local _output = model:forward(_input)
      _output = _output:float()
      local _err = criterionMSE:forward(_output, _target)
      _mse = _mse + _err
      local _err2 = math.log(_output[_label]) --criterionNLL:forward(_output, _label)
      nll_error = nll_error + _err2
      confusion:add(_output, _target)
   end
   _mse = math.sqrt(_mse/nTesting)
   nll_error = -1 * (nll_error/nTesting)
   -- timing
   time = sys.clock() - time
   time = time/nTesting
   print("==> time to test 1 sample = " .. (time*1000) .. 'ms')

-- print confusion matrix
   print(confusion.totalValid*100)

   print('====> epoch: ' .. epoch .. ', RMSE (test set) : ', _mse )
   print('====> epoch: ' .. epoch .. ', NLL (test set) : ', nll_error )
   print('')
   testRMSELogger:add{['RMSE (test set)'] = _mse}
   testNLLLogger:add{['NLL_ERROR (test set)'] = nll_error}
--   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save L1 filters to image file, just for funsies
   local weight_l1 = model.modules[1].modules[1].weight:float()
   local filters_l1 = {}
   for i=1,weight_l1:size(1) do
      table.insert(filters_l1, weight_l1[i]:view(math.sqrt(weight_l1:size(2)),math.sqrt(weight_l1:size(2))))
   end
   image.save(opt.results_path .. '/l1_' .. epoch .. '.png', image.toDisplayTensor{input=filters_l1,
                                                                      padding=3})
   image.save(opt.results_path .. '/l1color_' .. epoch .. '.png', image.toDisplayTensor{input=weight_l1, padding=3})

   if opt.save then
      -- save/log current net
      local filename = paths.concat(opt.results_path, 'model_' .. epoch .. '.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('<trainer> saving network to '..filename)
      print('')
      print('')
      -- save network to disk finally
      torch.save(filename, model)
   end
end
