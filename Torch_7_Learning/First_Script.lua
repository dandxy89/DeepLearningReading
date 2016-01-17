-- Dan Dixey
-- First Torch Script

-- Date Created: 8/4/2015

-- Starting Torch
-- th -lparallel -loptim -lpl -limage

-----------------------------------------------------------------------------

-- Creating First Array
my_table = {1, 2, 3}

-- Print Command
print(my_table)

-- Overwrite Object
my_table = {my_var = 'hello', my_other_var = 'world'}

-- Print Command
print(my_table)

-- Define a Function
my_function = function() print('hello world') end

-- Execute/Call Function
my_function()
------------------------------------------------------------------------------


----------------------------------------------------------------------------

--[[Multi-Layer Perceptron]]--

require 'dp' -- Import the dp module

------------------------------------------------------------------------------

--[[hyperparameters]]--

opt = {
   nHidden = 100, --number of hidden units
   nHidden2 = 50, -- number of hidden units on the second layer
   learningRate = 0.05, --training learning rate
   momentum = 0.9, --momentum factor to use for training
   maxOutNorm = 1, --maximum norm allowed for output neuron weights
   batchSize = 500, --number of examples per mini-batch
   maxTries = 100, --maximum number of epochs without reduction in validation error.
   maxEpoch = 10 --maximum number of epochs of training
}

------------------------------------------------------------------------------

--[[data]]--

datasource = dp.Mnist{input_preprocess = dp.Standardize()}

print("feature size: ", datasource:featureSize())

------------------------------------------------------------------------------

--[[Model]]--

model = dp.Sequential{
   models = {
      dp.Neural{
         input_size = datasource:featureSize(), 
         output_size = opt.nHidden, 
         transfer = nn.Tanh(),
         sparse_init = true
      },
      dp.Neural{
         input_size = opt.nHidden, 
         output_size = opt.nHidden2,
         transfer = nn.Tanh(),
         sparse_init = true
      },
      dp.Neural{
         input_size = opt.nHidden2, 
         output_size = #(datasource:classes()),
         transfer = nn.LogSoftMax(),
         sparse_init = true
      }
   }
}

------------------------------------------------------------------------------

--[[Propagators]]--
train = dp.Optimizer{
   loss = dp.NLL(),
   visitor = { -- the ordering here is important:
      dp.Momentum{momentum_factor = opt.momentum},
      dp.Learn{learning_rate = opt.learningRate},
      dp.MaxNorm{max_out_norm = opt.maxOutNorm}
   },
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   progress = true
}
valid = dp.Evaluator{
   loss = dp.NLL(),
   feedback = dp.Confusion(),  
   sampler = dp.Sampler()
}
test = dp.Evaluator{
   loss = dp.NLL(),
   feedback = dp.Confusion(),
   sampler = dp.Sampler()
}

------------------------------------------------------------------------------

--[[Experiment]]--
xp = dp.Experiment{
   model = model,
   optimizer = train,
   validator = valid,
   tester = test,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.maxTries
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

------------------------------------------------------------------------------

xp:run(datasource)
print(xp:report())

------------------------------------------------------------------------------