# Environment for studying the AWAKE Run 2 Electon Transfer Line

### Functionality for optimisation, plotting, error studies, steering, etc. 

## Optimiser:
* Enter the details of the parameters you want to optimise into x (in runOpt)
* The objective function is under step in OptEnv
* Most of the optimiser parameters are set under runOpt e.g number of particles to track, solver to use, width of foil at beam waist

## ErrorEnv allows for alignment and steering simulations:
- Adds current corrector values to new ones, so can string steering/alignment methods together
* SVD
* DFS
* One-to-one
* Quad shunting
	* Vary quad strength by 20% and measure at downstream BPM to estimate quad-beam offset
	* Correct for this offset, and iterate along beamline
	* Enter gain and number of iterations of quad shunting to perform
	* Max correction of 100 um
	* Switches higher order magnets on and off (self.switch_on_higher_order())
* Sextupole shunting
* Misc.: 
	* zeroCorr - zero all correctors
	* read_bpms

## Plotting (some of these may need some work)
* Phase vs s (plot.phase())
* Beam distributions (plotmat)
* Heatmap of beam size vs s for varying parameters (heatmap)
* Twiss for chromatic/detuning with amplitude effects (diffTwiss)
* Bar chart of different orders of contributions to beam size (MapTable)
* Layout of beamline (Survey)
* Modelling the plasma cell as quads (plasma_focus)