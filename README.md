# * Python code for analysis of electrophysiology data

## Requirements (modules):
   1. neo
   2. numpy
   3. scipy
   
## EphysClass: Captures and saves all information for a given Axon bionary file containing one or more sweeps

## Currently implemented routines:
   1. show:  displays selected channels and sweeps in a multipanel matplotlib window
   2. info:  provides meaningful information related to the Ephys object defined by a given Ephys file
   3. seriesres_currentclamp:	 measures the voltage jump associated with a current step to calculate the series resistance


## Todo list

   - [] Infer number of stimulation pulses from the trigger channel
   - [] Infer ISI of stimulation from the trigger channel
   - [] Detect peak amplitude of a response from the response channel
   - [] Calculate delay in peak response w.r.t. stimulation time
   - [] Calculate risetime of response