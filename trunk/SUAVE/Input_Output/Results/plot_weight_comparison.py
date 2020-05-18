import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from SUAVE.Core import Units
from matplotlib.pyplot import cm


def plot_weight_comparison(lst_files, caseNames=[], units='kg'):
    if len(caseNames) < len(lst_files):
        warnings.warn('caseNames and the number of files do not match, adding generic case names')
        for i in range(len(caseNames), len(lst_files)):
            caseNames.append("Case " + str(i))
    if not isinstance(lst_files, list):
        lst_files = [lst_files]
    main_weight = {}
    sub_weight = {}
    lst_cases = []
    lst_subsystems = ["Structural weight", "Propulsion weight",
                      "System weight", "Operational items weight",
                      "Payload weight"]
    lst_n_systems = [8, 4, 9, 3, 3]
    lst_dicts = [{} for _ in range(len(lst_subsystems))]
    for j, file in enumerate(lst_files):
        data = pd.read_csv(file)
        weight_names = data.iloc[:, 0]
        for i in range(1, len(data.columns)):
            lst_cases.append(caseNames[j] + " " + data.columns[i])
            find_idx_main_weight(data.iloc[:, i], weight_names, main_weight)
            find_idx_sub_weight(data.iloc[:, i], weight_names, sub_weight, lst_subsystems)
            for k, sub in enumerate(lst_subsystems):
                find_idx_subsystem_weight(data.iloc[:, i], weight_names,
                                          lst_dicts[k], sub, lst_n_systems[k])
    plot_histogram('Main weight distribution', units, lst_cases, main_weight)
    plot_histogram('Subsystem weight distribution', units, lst_cases, sub_weight, (10, 5))
    for k, sub in enumerate(lst_subsystems):
        if lst_n_systems[k] > 4:
            plot_histogram([sub + ' breakdown'], units, lst_cases, lst_dicts[k], (12, 5))
        else:
            plot_histogram([sub + ' breakdown'], units, lst_cases, lst_dicts[k], (12, 5))


def find_idx_main_weight(col, weight_names, dict):
    for i, name in enumerate(weight_names):
        if name == "Zero Fuel Weight":
            if not name in dict:
                dict[name] = [col[i]]
            else:
                dict[name].append(col[i])
        if name == "Empty Weight":
            if not name in dict:
                dict[name] = [col[i]]
            else:
                dict[name].append(col[i])
        if name == "Operational Empty Weight":
            if not name in dict:
                dict[name] = [col[i]]
            else:
                dict[name].append(col[i])


def find_idx_sub_weight(col, weight_names, weight_dict, lst_subsystems):
    for i, name in enumerate(weight_names):
        if name in lst_subsystems:
            if not name in weight_dict:
                weight_dict[name] = [col[i]]
            else:
                weight_dict[name].append(col[i])





def find_idx_subsystem_weight(col, weight_names, weight_dict, subsystem, n_systems):
    for i, name in enumerate(weight_names):
        if name == subsystem:
            start = 1
            for j in range(i + start, i + n_systems + 1):
                if not weight_names[j] in weight_dict:
                    weight_dict[weight_names[j]] = [col[j]]
                else:
                    weight_dict[weight_names[j]].append(col[j])


def plot_histogram(title, units, lst_cases, weight_dict, size=None):
    total_bars = len(lst_cases) * len(weight_dict)
    if units == 'kg':
        units_factor = 1
        units_label = 'Weights (kg)'
    elif units == 'lb' or units == 'lbs':
        units_factor = 1 * Units.kg / Units.lbs
        units_label = 'Weights (lbs)'
    else:
        raise ValueError("Unit not implimented")
    if size is not None:
        fig = plt.figure(figsize=size)
    else:
        fig = plt.figure()
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.grid(which='major', axis='y', linestyle='-', linewidth=1)
    ax.grid(which='minor', axis='y', linestyle='-', linewidth=.5)
    left, right = plt.xlim()
    bar_width = 0.75 / total_bars
    colors = cm.Paired(np.linspace(0,1,10))
    rs = [np.linspace(0,1,len(weight_dict))]
    for i in range(len(lst_cases) - 1):
        rs.append([x + bar_width for x in rs[-1]])
    col = 0
    for k, v in weight_dict.items():
        for i, name in enumerate(lst_cases):
            if col == 0:
                ax.bar(rs[i][col], v[i] * units_factor, label=name, width=bar_width, color=colors[i], edgecolor='white')
            else:
                ax.bar(rs[i][col], v[i] * units_factor, width=bar_width, color=colors[i], edgecolor='white')

        col += 1
    plt.ylabel(units_label)
    rs = np.array(rs)
    diff = (np.amax(rs, axis=0) - np.amin(rs, axis=0)) / 2.
    plt.xticks(list(rs[0, :] + diff), weight_dict.keys())
    plt.legend()
    fig.tight_layout()
