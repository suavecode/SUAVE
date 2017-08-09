## How to Read (and Write) the Documentation

This is an explanation of how documentation is built in SUAVE so that the user can know what to look for and how to write it if they wish to add their own code.

### Docstrings

All classes and functions in SUAVE have docstrings have docstrings. These give the user an understanding of what the function does and information on the input and output variables. 

#### Classes

For a class, the docstring has three parts:

* General description
* Assumptions
* Source

The general description provides a brief summary of what the class is used for. Assumptions list any significant assumptions are important in how it is used. This may be listed as None or N/A if there are no important assumptions. The source section should provide the general source of methods in the class. If there is no overall source, a message like 'see sources in functions' may be appropriate. There are also cases where a source is not relevant, such as simple geometric functions, and may also be listed as None or N/A.

#### Class Docstring Template

This is the general template that can be used when creating a new class. It should be placed directly under the class declaration.

    """<Description>
    
    Assumptions:
    <any assumptions>
    
    Source:
    <source>
    """
    
#### Functions

For functions there are six categories:

* Description

This is a general description of what the function does. It should also include any key information that does not fit into one of the other categories.

* Assumptions

This should contain any assumptions made by the function. None or N/A can be used if none are used.

* Source

The source of any methods that have been implemented. Simple methods like basic geometric relations do not need a source.

* Inputs

This should contain any variables or functions passed to the function. If the passed variable is a data structure, the components that are used should be listed. Each item should include a description if it is not obvious from the name, and any relevant units. It may 

### Example Case

This is the docstring from [Supersonic_Nozzle]()

        """ This computes the output values from the input values according to
        equations from the source.
        
        Assumptions:
        Constant polytropic efficiency and pressure ratio
        
        Source:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
        
        Inputs:
          conditions data class with conditions.freestream.
            isentropic_expansion_factor         [Unitless]
            specific_heat_at_constant_pressure  [J/(kg K)]
            pressure                            [Pa]
            stagnation_pressure                 [Pa]
            stagnation_temperature              [K]
            universal_gas_constant              [J/(kg K)] (this is a bad name)
            mach_number                         [Unitless]
                
          self.inputs.
            stagnation_temperature              [K]
            stagnation_pressure                 [Pa]
                   
        Outputs:
          self.outputs.
            stagnation_temperature              [K]  
            stagnation_pressure                 [Pa]
            stagnation_enthalpy                 [J/kg]
            mach_number                         [Unitless]
            static_temperature                  [K]
            static_enthalpy                     [J/kg]
            velocity                            [m/s]
            static_pressure                     [Pa]
            area_ratio                          [Unitless]
                
        Properties Used:
          self.
            pressure_ratio                      [Unitless]
            polytropic_efficiency               [Unitless]
            """       
            
The docstring is broken into 