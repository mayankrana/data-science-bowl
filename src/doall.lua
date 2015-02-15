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
cmd:option('-save',            false,       'save models')
cmd:option('-log',             true,        'save log')
cmd:option('-plot',            false,       'save plot for error')
cmd:option('-results_path',    'results/exp_3_3_hidden_1024',   'subdirectory to save/log experiments in')
cmd:option('-learningRate',    10e-2,        'learning rate at t=0')--5e-2
cmd:option('-momentum',        0.6,         'momentum')--0.6
cmd:option('-weightDecay',     1e-5,        'weight decay')--1e-5
cmd:option('-batchSize',       64,          'mini-batch size (1 = pure stochastic)')
cmd:option('-progressBar',     true,        'Display a progress bar')
cmd:option('-dataTest',        false,       'visual sanity checks for data loading')
cmd:option('-dropout',         true,       'do dropout with 0.5 probability')
cmd:option('-retrain',         "none",      'provide path to model to retrain with')
cmd:text()
opt = cmd:parse(arg or {})

-- Number of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
cutorch.setDevice(opt.gpuid)

sampleSize = {1, 48, 48}

dataRoot = "../data/"

lightTesting = true

--Number of images to be used for each epoch
epochSize = opt.batchSize * 1000
--Max number of epochs to run the experiment for
--Keep it higher and kill the process upon convergence
maxEpochs = 100

dofile('1_data.lua')
if not opt.dataTest then
   dofile('2_model_size_48_2_hidden_1024.lua')
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

      --Save train and test score to log files

      if opt.plot then
      	 local filename = paths.concat(opt.results_path, 'train.plot')
	 os.execute('mkdir -p ' .. sys.dirname(filename))
	 print('Saving training plot to '..filename)
	 trainLogger:style{['% nll error (train set)'] = '-'}
	 trainLogger:plot()

	 filename = paths.concat(opt.results_path, 'test.plot')
	 print('Saving test plot to '..filename)
	 testLogger:style{['% nll error (train set)'] = '-'}
	 testLogger:plot()
      end
--      if epoch == 50 then lightTesting = false; end
   end
end
