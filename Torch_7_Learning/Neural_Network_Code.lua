-- Script:      Importing the Bank Marketing Dataset
-- Creator:     Daniel Dixey
-- Started:     9 April 2015
-- Modified:    13 April 2015

-----------------------------------------------------------------------------
-- Importing Modules into Lua/Torch
require("torch" )    -- Import Torch Module
require("paths" )    -- Directory Navigation
require("nn" )       -- Neural Networking Module
require("dp")        -- dp is a deep learning library 

-----------------------------------------------------------------------------
-- User Defined Functions

local function import_and_convert(Name )

    local csv = require("csv" )
 
    local inputs, labels = {}, {}
 
    local f, i = csv.open(Name .. ".csv"), 0
    for fields in f:lines() do
         if i > 0 then -- skip header
            inputs[i] = {unpack(fields, 1, 63)}
            labels[i] = fields[64]
         end
         i = i + 1
    end
    
    local dataset = {}

    for i = 1, i-1 do
       dataset[i] = {inputs[i], torch.Tensor{labels[i]}}
    end   
    
    -- Save Datasource
    torch.save(Name .. "_Data.th7", dataset )
    
    -- Return Data to Source
    return dataset
end

local function Importing_Data()
    
    print('============================================================')
    print('Constructing dataset')
    print('')

    -- Check if Torch Files Exist
    if paths.filep("Testing_Data.th7" ) and paths.filep("Training_Data.th7" ) then
        
        print("Importing Pre-Made Tensor Files" )
        Training_Data   = torch.load("Training_Data.th7" )
        Testing_Data    = torch.load("Testing_Data.th7" )
        
        print("Imported and Created Tensor Files" )
        
    else
        Training_Data = import_and_convert("Training" )
        Testing_Data = import_and_convert("Testing" )
        
        print("Imported and Created Tensor Files" )
        
    end

    -- Return data to Main
    return Training_Data, Testing_Data
end

-- Creating the Neural Network
local function NN_model()
    
    -- Define Hyperparameters
    opt = {
       nHidden = {100}, --number of hidden units
       learningRate = 0.1, --training learning rate
       momentum = 0.9, --momentum factor to use for training
       maxOutNorm = 1, --maximum norm allowed for output neuron weights
       batchSize = 128, --number of examples per mini-batch
       maxTries = 100, --maximum number of epochs without reduction in validation error.
       maxEpoch = 1000 --maximum number of epochs of training
    }

    -- Create Neural Network Model
    model = dp.Sequential{
           models = {
                  dp.Neural{
                         input_size = 63, 
                         output_size = opt.nHidden[1], 
                         transfer = nn.ReLU(),
                         sparse_init = false
                      },
                  dp.Neural{
                         input_size = opt.nHidden[1], 
                         output_size = 1,
                         transfer = nn.LogSoftMax(),
                         sparse_init = false
                      }
                   }            
                }

    -- Define the Propagators
    train = dp.Optimizer{
                       loss = dp.NLL(),
                       visitor = { -- the ordering here is important:
                                  dp.Momentum{momentum_factor = opt.momentum},
                                  dp.Learn{learning_rate = opt.learningRate},
                                  dp.MaxNorm{max_out_norm = opt.maxOutNorm}
                                   },
                       feedback = dp.Confusion(),
                       sampler = dp.Sampler{},
                       progress = true
                                    }
                       test = dp.Evaluator{
                                    loss = dp.NLL(),
                                    feedback = dp.Confusion(),
                                    sampler = dp.Sampler{}
                        }

    --Define the Hierarchy Of Evaluations
    xp = dp.Experiment{
                       model = model,
                       optimizer = train,
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

    -- Return data to Main
    return model, train, xp
end    

-----------------------------------------------------------------------------
-- Main Function -Adopting a Similar Approach as Python
local function main()
    
    -- Start the Stop Watch
    start_time = os.clock();

    -- Run the Import Data Function
    Training_Data, Testing_Data = Importing_Data()
    
    -- Creating the Model    
    model, train, xp = NN_model()

    -- Print Time to Process
    diff = os.clock() - start_time;
    print(string.format('Script took %.5f seconds\n', diff));

end

-----------------------------------------------------------------------------
-- Run the Main Function
main()
