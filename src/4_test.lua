require 'torch'
require 'xlua'
require 'optim'
require 'image'

print '=> 4_test.lua'
print '<4_test.lua>: Defining test procedure'
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

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
      -- test sample
      local input, target = getTest(t, lightTesting)
      input = input:cuda()
      local output = model:forward(input)
      output = output:float()
      local err = criterion:forward(output, target)
      nll_error = nll_error + err
   end
   nll_error = nll_error/nTesting
   -- timing
   time = sys.clock() - time
   time = time/nTesting
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   print('==> epoch: ' .. epoch .. ', logloss (test set) : ' .. nll_error )
   testLogger:add{['logloss (test set)'] = nll_error}

   -- save/log current net
   local filename = paths.concat(opt.save, 'model_' .. epoch .. '.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('<trainer> saving network to '..filename)
   print('')
   print('')
   -- save L1 filters to image file, just for funsies
   local weight_l1 = model.modules[1].modules[1].weight:float()
   local filters_l1 = {}
   for i=1,weight_l1:size(1) do
      table.insert(filters_l1, weight_l1[i]:view(math.sqrt(weight_l1:size(2)),math.sqrt(weight_l1:size(2))))
   end
   image.save('results/l1_' .. epoch .. '.png', image.toDisplayTensor{input=filters_l1,
                                                                      padding=3})
   image.save('results/l1color_' .. epoch .. '.png', image.toDisplayTensor{input=weight_l1, padding=3})
   -- save network to disk finally
   torch.save(filename, model)
end
