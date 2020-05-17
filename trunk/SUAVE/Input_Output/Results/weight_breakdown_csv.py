import pandas as pd
import os.path
import datetime
import numpy as np


def weight_breakdown_to_csv(config, filename='weight_comparison.csv', header=None):
    weight_breakdown = config.weight_breakdown
    mass_properties = config.mass_properties
    ac_type = config.systems.accessories
    if not os.path.isfile(filename):
        lst_weights = ['Maximum Takeoff Weight', 'Fuel Weight', 'Zero Fuel Weight',
                       'Operating Empty Weight', 'Empty Weight']
        lst_weights.append('Structural weight')
        for tag, value in weight_breakdown.structures.items():
            if tag != "total":
                tag = tag.replace('_', ' ')
                string = tag[0].upper() + tag[1:]
                lst_weights.append(string)
        lst_weights.append('Propulsion weight')
        for tag, value in weight_breakdown.propulsion_breakdown.items():
            if tag != "total":
                tag = tag.replace('_', ' ')
                string = tag[0].upper() + tag[1:]
                lst_weights.append(string)
        lst_weights.append('System weight')
        for tag, value in weight_breakdown.systems_breakdown.items():
            if tag != "total":
                tag = tag.replace('_', ' ')
                string = tag[0].upper() + tag[1:]
                lst_weights.append(string)
        lst_weights.append('Operational items weight')
        for tag, value in weight_breakdown.operational_items.items():
            if tag != "total":
                tag = tag.replace('_', ' ')
                string = tag[0].upper() + tag[1:]
                lst_weights.append(string)
        lst_weights.append('Payload weight')
        for tag, value in weight_breakdown.payload_breakdown.items():
            if tag != "total":
                tag = tag.replace('_', ' ')
                string = tag[0].upper() + tag[1:]
                lst_weights.append(string)
        df = pd.DataFrame(lst_weights, columns=["Weights (kg)"])
        df.to_csv(filename, index=False)

    if header == None:
        header = datetime.datetime.now().strftime(" %A, %d. %B %Y %I:%M:%S %p")
    lst_weights = [mass_properties.max_takeoff, weight_breakdown.fuel, weight_breakdown.zero_fuel_weight, \
                   weight_breakdown.operating_empty, weight_breakdown.empty]

    lst_weights.append(weight_breakdown.structures.total)
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
