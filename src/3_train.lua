require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '=> 3_train.lua'
print '<3_train.lua>: Defining some tools'

trainLogger = optim.Logger(paths.concat(opt.results_path, 'train.log'))

if model then
   if opt.retrain ~= "none" then
      local parameters,gradParameters = model:getParameters()
      local mod2 = torch.load(opt.retrain):float()
      local p2,gp2 = mod2:getParameters()
      parameters:copy(p2)
      gradParameters:copy(gp2)
   end
   model:cuda()
   parameters,gradParameters = model:getParameters()
   collectgarbage()
end

print '<3_train.lua>: configuring optimizer'

optimState = {
   learningRate = opt.learningRate,
   weightDecay = opt.weightDecay,
   momentum = opt.momentum,
   learningRateDecay = 5e-7
}
optimMethod = optim.sgd

print '<3_train.lua>: Defining training procedure'
function train()

   -- epoch tracker
   epoch = epoch or 1
   -- local vars
   local time = sys.clock()
   local batchSize = opt.batchSize
   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   local nll_error = 0
   for t = 1,epochSize,batchSize do
      -- disp progress
      if opt.progressBar then xlua.progress(t, epochSize) end

      -- create mini batch
      local inputs, targets = getBatch(batchSize)
      inputs = inputs:cuda()

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- f is the average of all criterions for a batch
         local f = 0;

         -- evaluate function for complete mini batch
         -- estimate f
         local outputs = model:forward(inputs)
         outputs = outputs:float()
         local df_dw = torch.Tensor(outputs:size(1), outputs:size(2))
         for i=1,batchSize do
            -- estimate MSE individually per branch
            criterion:forward(outputs[i], targets[i])
            -- estimate df/dW
            df_dw[i] = criterion:backward(outputs[i], targets[i])
            local err = criterion:forward(outputs[i], targets[i])
            -- sum individual RMSE
            nll_error = nll_error + err
            f = f + err
         end
         model:backward(inputs, df_dw:cuda())
         -- normalize gradients and f(X)
         gradParameters:div(batchSize)
         -- fgradParameters:mul(#branch)
         f = f/batchSize

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      optim.sgd(feval, parameters, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time/epochSize
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   nll_error = nll_error/epochSize
   print('===>epoch: ' .. epoch .. ', logloss (train set): ' .. nll_error)
   print('')
   print('')
   trainLogger:add{['logloss (train set)'] = nll_error}

end
