-- Script:      Importing the Bank Marketing Dataset
-- Creator:     Daniel Dixey
-- Started:     9 April 2015
-- Modified:    13 April 2015

-----------------------------------------------------------------------------
-- Importing Modules into Lua/Torch
require("torch" )    -- Import Torch Module
require("paths" )    -- Directory Navigation
require("csvigo" )   -- Import CSV Files
require("nn" )       -- Neural Networking Module
require("dp")        -- dp is a deep learning library 

-----------------------------------------------------------------------------
-- User Defined Functions
local function import_data(Name)
    -- Import Data into an Object    
    csvdata = csvigo.load{path= Name .. ".csv", mode='tidy', verbose=false}
    

end

-----------------------------------------------------------------------------
-- Main Function -Adopting a Similar Approach as Python
local function main()
    -- Import the Training Dataset
    import_data('Training')
end

-----------------------------------------------------------------------------
-- Run the Main Function
main()