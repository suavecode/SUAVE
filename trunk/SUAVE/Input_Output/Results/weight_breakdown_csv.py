import pandas as pd
import os.path
import datetime
import numpy as np


def weight_breakdown_to_csv(config, filename='weight_comparison.csv', header=None):
    """ Function that prints the weight breakdown of config to a csv named appropriately

        Source:
            N/A
       Inputs:
            config - data dictionary with vehicle properties
                .weight_breakdown   - data dictionary with the weight breakdown of the vehicle
                .mass_properties    - main mass properties of the aircraft, includes MTOW, ZFW, OEW and EW
       Outputs:
            N/A
        Properties Used:
            N/A
    """
    weight_breakdown    = config.weight_breakdown
    mass_properties     = config.mass_properties
    if not os.path.isfile(filename):
        lst_weights = ['Maximum Takeoff Weight', 'Fuel Weight', 'Zero Fuel Weight', 'Operating Empty Weight',
                       'Empty Weight', 'Structural weight']
        for tag, value in weight_breakdown.structures.items():
            if tag != "total":
                tag     = tag.replace('_', ' ')
                string  = tag[0].upper() + tag[1:]
                lst_weights.append(string)
        lst_weights.append('Propulsion weight')
        for tag, value in weight_breakdown.propulsion_breakdown.items():
            if tag != "total":
                tag     = tag.replace('_', ' ')
                string  = tag[0].upper() + tag[1:]
                lst_weights.append(string)
        lst_weights.append('System weight')
        for tag, value in weight_breakdown.systems_breakdown.items():
            if tag != "total":
                tag     = tag.replace('_', ' ')
                string  = tag[0].upper() + tag[1:]
                lst_weights.append(string)
        lst_weights.append('Operational items weight')
        for tag, value in weight_breakdown.operational_items.items():
            if tag != "total":
                tag     = tag.replace('_', ' ')
                string  = tag[0].upper() + tag[1:]
                lst_weights.append(string)
        lst_weights.append('Payload weight')
        for tag, value in weight_breakdown.payload_breakdown.items():
            if tag != "total":
                tag     = tag.replace('_', ' ')
                string  = tag[0].upper() + tag[1:]
                lst_weights.append(string)
        df = pd.DataFrame(lst_weights, columns=["Weights (kg)"])
        df.to_csv(filename, index=False)

    if header is None:
        header = datetime.datetime.now().strftime(" %A, %d. %B %Y %I:%M:%S %p")
    lst_weights = [mass_properties.max_takeoff, weight_breakdown.fuel, weight_breakdown.zero_fuel_weight,
                   weight_breakdown.operating_empty, weight_breakdown.empty, weight_breakdown.structures.total]

    for tag, value in weight_breakdown.structures.items():
        if tag != "total":
            if isinstance(value, np.ndarray):
                lst_weights.append(value[0])
            else:
                lst_weights.append(value)
    lst_weights.append(weight_breakdown.propulsion_breakdown.total)
    for tag, value in weight_breakdown.propulsion_breakdown.items():
        if tag != "total":
            if isinstance(value, np.ndarray):
                lst_weights.append(value[0])
            else:
                lst_weights.append(value)
    lst_weights.append(weight_breakdown.systems_breakdown.total)
    for tag, value in weight_breakdown.systems_breakdown.items():
        if tag != "total":
            if isinstance(value, np.ndarray):
                lst_weights.append(value[0])
            else:
                lst_weights.append(value)
    lst_weights.append(weight_breakdown.operational_items.total)
    for tag, value in weight_breakdown.operational_items.items():
        if tag != "total":
            if isinstance(value, np.ndarray):
                lst_weights.append(value[0])
            else:
                lst_weights.append(value)
    lst_weights.append(weight_breakdown.payload_breakdown.total)
    for tag, value in weight_breakdown.payload_breakdown.items():
        if tag != "total":
            if isinstance(value, np.ndarray):
                lst_weights.append(value[0])
            else:
                lst_weights.append(value)
    df = pd.read_csv(filename)
    while header in df.columns:
        header = header + "1"
    df[header] = lst_weights
    df.to_csv(filename, index=False)
