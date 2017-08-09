##Energy Networks
Energy networks are really a fancy name for your propulsion system. The reason we call them a network rather than "engine" or the like is that it encompasses far more than that. These "energy networks" are the complex systems that future aircraft will incorporate that can reduce or eliminate fuel burn. For example if you have a hybrid gas-electric aircraft you don't have just an engine. You have an engine, a motor, batteries, a generator, a gearbox, a propeller or ducted fan, wiring, electronics... and that doesn't include the parts inside the internal combustion engine. Modeling these systems completely is essential in designing such a vehicle.

To do this we depart from the typical SUAVE structure of analyses and methods. This was done because we want to look at each component of the network individually. All components behave in their own ways. The purpose of the network is to link together every component in the system to work together.

### File Structure
One of the biggest sources of confusion for energy networks is the file structure. The files for every part of a network are located at trunk/SUAVE/**Components/Energy**. Within that we have several subfolders.

#### /Converters
Converters are defined as component that takes energy or power from one form to another. It could be electrical to mechanical, or even chemical to mechanical, etc.. Examples in this folder include a motor, a solar panel, and the compressor section of a jet engine.
#### /Distributors
Distributors move power from one part of the network to another. The common use of this is an electronic speed controller for a small UAV.
#### /Networks
This is where all the "Network" scripts that tie together all of the pieces are kept. Examples in here are a turbofan network.
#### /Peripherals
Peripherals are items that rely on the network but do not produce thrust or power for the vehicle. For example, avionics require electricity which must be accounted for.
#### /Processes
Processes are non tangible parts of a network that are necessary. For example the process is thrust. This function is useful for jet engines to combine and dimensionalize the final thrust once all the components are combined in the network.
#### /Storages
Storages, for now, include batteries. However, in the future we could have fuel tanks here. Currently fuel is only handled as a mass variation. 

### Component Example
Below is a simple example of the most basic energy component in SUAVE. All energy components are classes. The  A solar panel is a converter since it converts a photonic flux calculated by a solar radiation model and converts it to a power.

	# Solar_Panel.py
	#
	# Created:  Jun 2014, E. Botero
	# Modified: Jan 2016, T. MacDonald

	# ----------------------------------------------------------------------
	#  Imports
	# ----------------------------------------------------------------------
	
	# suave imports
	import SUAVE
	
	from SUAVE.Components.Energy.Energy_Component import Energy_Component
	
	# ----------------------------------------------------------------------
	#  Solar_Panel Class
	# ----------------------------------------------------------------------
	class Solar_Panel(Energy_Component):
	    
	    def __defaults__(self):
	        self.area       = 0.0
	        self.efficiency = 0.0
	    
	    def power(self):
	        
	        # Unpack
	        flux       = self.inputs.flux
	        efficiency = self.efficiency
	        area       = self.area
	        
	        p = flux*area*efficiency
	        
	        # Store to outputs
	        self.outputs.power = p
	    
	        return p

These classes contain functions that simulate a process. Multiple functions can exist within a component. This component has defaults, inputs, and outputs. Defaults are provided that give the user an idea of what the fixed parameters of the component are. These values are set when initializing a vehicle. This is the recipe that all energy components are built off. 

### Network Scripts 
Network scripts are the link between these components. The network script allows you to reconfigure the connection between components to create your dream propulsion system. These must be logically created as components have set inputs and outputs. For example, you can't magically go from a battery to a propeller without anything in between. Some knowledge of the inputs and outputs are necessary, however they're generally quite intuitive. One interesting avenue for SUAVE is that you can create networks of networks.

The linking process works like this:

        # step 1
        solar_flux.solar_radiation(conditions)
        # link
        solar_panel.inputs.flux = solar_flux.outputs.flux
        # step 2
        solar_panel.power()
        # link
        solar_logic.inputs.powerin = solar_panel.outputs.power
        
Notice the first step above is to calculate the solar radiation. Once the solar radiation is calculated the components are linked and step 2 can continue with the the power being calculated. From there a solar logic component will use that power.

The other main hallmark of a network is that they are called at every point in the mission to calculate the state of the system. Given some *conditions* data that defines the state of the vehicle the components must provide back to the mission being solved a thrust force and a mass rate. Other outputs can be stored back to conditions, however a thrust and a mass rate must be returned.

### Vehicle Script Setup of a Network

Here we will provide a snippet of the turbofan setup for a B737. 

    # ------------------------------------------------------------------
    #  Component 3 - Low Pressure Compressor

    # instantiate 
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = 'low_pressure_compressor'

    # setup
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio        = 1.14    

    # add to network
    turbofan.append(compressor)
    
In the above example, a compressor is added to the network. The compressor is tagged as the low_pressure_compressor to distinguish it from the high pressure compressor. The polytropic efficiency and pressure ratio are set. Finall it is appended to the network.
