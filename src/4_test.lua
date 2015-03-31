require 'torch'
require 'xlua'
require 'optim'
require 'image'

print '=> 4_test.lua'
print '<4_test.lua>: Defining test procedure'
testLogger = optim.Logger(paths.concat(opt.results_path, 'test.log'))

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(class_id_to_name)

-- test function
function test()
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   -- test over test data
   print('<4_test.lua>: Testing on test set:')
   local nll_error = 0
   for t = 1,nTesting do
      if opt.progressBar then xlua.progress(t, nTesting) end
      local err = 0
      for it = 1,5 do
      	  -- test sample
	  local input, target = getTest(t, lightTesting)
	  input = input:cuda()
      	  local output = model:forward(input)
      	  output = output:float()
      	  err = err + criterion:forward(output, target)
	  confusion:add(output, target)
      end
      nll_error = nll_error + err/5
--      confusion:add(output, target)
   end
   nll_error = nll_error/nTesting
   -- timing
   time = sys.clock() - time
   time = time/nTesting
   print("==> time to test 1 sample = " .. (time*1000) .. 'ms')

-- print confusion matrix
   print(confusion.totalValid*100)

   print('====> epoch: ' .. epoch .. ', logloss (test set) : ', nll_error )
   print('')
   testLogger:add{['logloss (test set)'] = nll_error}
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

   --save model at every 10th epoch
   if (opt.save and math.fmod(epoch,30)==0) then
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
