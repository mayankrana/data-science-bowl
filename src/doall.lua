require 'torch'
require 'cutorch'

torch.setdefaulttensortype('torch.FloatTensor')
cmd = torch.CmdLine()
cmd:text()
cmd:text('DataScienceBowl Training script')
cmd:text()
cmd:text('Options:')
cmd:option('-seed',            1,           'fixed input seed for repeatable experiments')
cmd:option('-threads',         1,           'number of threads')
cmd:option('-gpuid',           1,           'gpu id')
cmd:option('-save',            'results',   'subdirectory to save/log experiments in')
cmd:option('-learningRate',    5e-2,        'learning rate at t=0')
cmd:option('-momentum',          0.6,        'momentum')
cmd:option('-weightDecay',       1e-5,        'weight decay')
cmd:option('-batchSize',       32,           'mini-batch size (1 = pure stochastic)')
cmd:option('-progressBar',     true,       'Display a progress bar')
cmd:option('-dataTest',     false,       'visual sanity checks for data loading')
cmd:option('-dropout',     false,       'do dropout with 0.5 probability')
cmd:option('-retrain',     "none",       'provide path to model to retrain with')
cmd:text()
opt = cmd:parse(arg or {})

-- Number of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
cutorch.setDevice(opt.gpuid)

sampleSize = {1, 90, 90}

dataRoot = "../data/"

lightTesting = true

--Number of images to be used for each epoch
epochSize = opt.batchSize * 1000
--Max number of epochs to run the experiment for
--Keep it higher and kill the process upon convergence
maxEpochs = 20

dofile('1_data.lua')
if not opt.dataTest then
   dofile('2_model.lua')
   dofile('3_train.lua')
   dofile('4_test.lua')

   epoch = 0
   --get the initial error on random weights wthout training
   test()
   while (epoch < maxEpochs) do
      epoch = epoch + 1
      collectgarbage()
      train()
      collectgarbage()
      test()
--      if epoch == 50 then lightTesting = false; end
   end
end
