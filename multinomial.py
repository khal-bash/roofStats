from IPython.core.pylabtools import figsize
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax, var

def _outer_addition(vec1: np.array, vec2: np.array):

    if(vec1.ndim != 1 or vec2.ndim != 1 or vec1.shape != vec2.shape):
        print("Invalid vectors passed into outer_addition")
        return

    matrix1 = np.tile(vec1, (vec1.size, 1))
    matrix2 = np.tile(vec2, (vec2.size, 1)).T

    result = matrix1 + matrix2
    return result

def _compute_pmf(rusted, light, unrusted, step):

    total = rusted + light + unrusted

    p_rusted = np.arange(0, 1.000, step, dtype=np.float64)
    p_light = np.arange(0, 1.000, step, dtype=np.float64)
    
    p_clean = 1 - _outer_addition(vec1=p_rusted, vec2=p_light)
    p_clean[p_clean < 0] = 0

    exp_p_rusted = np.power(p_rusted, rusted)
    exp_p_light = np.power(p_light, light)
    exp_p_clean = np.power(p_clean, unrusted)

    powers = np.outer(exp_p_light, exp_p_rusted) * exp_p_clean

    coef = (factorial(total)
            /(factorial(rusted)*factorial(light)*factorial(unrusted)))

    result = coef*powers

    magnitude = np.sum(result)
    magnitude *= step
    norm_result = result / magnitude
    return norm_result

def _sum_over_axes(norm_pmf : np.array):
    ones_col = np.ones(shape=(norm_pmf.shape[1], 1))
    row_sums = np.matmul(norm_pmf, ones_col) #lightly rusted

    ones_row = ones_col.T[0]
    col_sums = np.matmul(ones_row, norm_pmf) #rusted

    return row_sums, col_sums

def _graph_pmf(norm_pmf : np.array, x_axis : np.array, binomial, _color = 'r',
               var1Label = 'Rusted', var2Label="Lightly Rusted"):

    light_rusted, rusted = _sum_over_axes(norm_pmf)
    
    plt.plot(x_axis, rusted.T, color=_color, label=var1Label)
    if(binomial):
        return

    plt.plot(x_axis, light_rusted, color='b', label=var2Label)
    

def _add_until_threshold(data_series : np.array, threshold : float):
    
    maximum = np.argmax(data_series)

    if(data_series[maximum] < 0.0001):
        return

    running = 0
    deviation = 0

    while running < (threshold * np.sum(data_series)):

        if(deviation != 0):

            if(maximum+deviation <= np.size(data_series) - 1):
                running += data_series[maximum + deviation]
            
            if(maximum-deviation >= 0):
                running += data_series[maximum - deviation]

        else:
            running += data_series[maximum]

        deviation += 1
    
    return max(maximum - deviation, 0), min(maximum + deviation,
                                            np.size(data_series)-1)

def _clean_x_points(x_points, force_binomial):

    x_ticks = [0]
    x_labels = [0]

    if(force_binomial):
        x_ticks.extend(x_points[3:])
        x_labels.extend(x_points[3:])
    else:
        x_ticks.extend(x_points)
        x_labels.extend(x_points)
        
    x_ticks.append(1)
    x_labels.append(1)

    if(x_ticks[1] == 0):
        x_ticks.pop(0)

    if(x_labels[1] == 0):
        x_labels.pop(0)

    if(x_ticks[-2] == 1):
        x_ticks.pop(-1)

    if(x_labels[-2] == 1):
        x_labels.pop(-1)

    x_labels = [round(num, 2) for num in x_labels]

    return x_ticks, x_labels

def _adjust_visuals(x_points, force_binomial):

    ax = plt.gca()
    ax.get_yaxis().set_visible(False)

    x_ticks, x_labels = _clean_x_points(x_points, force_binomial)

    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.xlabel('Percentage of Roof')
    plt.legend(loc='upper right')
    
def _manual_integration(norm_pmf : np.ndarray, force_binomial, color = 'r'):
    
    light_rusted, rusted = _sum_over_axes(norm_pmf)
    light_min, light_max = _add_until_threshold(light_rusted, 0.8)
    rust_min, rust_max = _add_until_threshold(rusted, 0.8)

    x_points = np.array([light_min,
                         np.argmax(light_rusted),
                         light_max,
                         rust_min,
                         np.argmax(rusted),
                         rust_max], dtype=float) * 0.001

    y_points = np.array([light_rusted[light_min].item(),
                         np.max(light_rusted),
                         light_rusted[light_max].item(),
                         rusted[rust_min].item(),
                         np.max(rusted),
                         rusted[rust_max].item()], dtype=float)

    if (force_binomial):
        y_points[0] = 0
        y_points[1] = 0
        y_points[2] = 0
   
    _colors_ = ['b', 'b', 'b', color, color, color]

    plt.vlines(x_points,
               np.zeros(shape = (1,4)),
               y_points,
               colors=_colors_,
               linestyles='dashed')
    
    return x_points

def compute_probabilities(var1,
                          var2,
                          remainder,
                          var1Label,
                          var2Label,
                          force_binomial=False,
                          color = 'r'):

    #resolution of results - smaller step is higher resolution
    # but more expensive
    step = 0.001

    pmf = _compute_pmf(var1,
                       var2,
                       remainder,
                       step)
                       
    x_axis = np.arange(0, 1, step, dtype=float)
    _graph_pmf(pmf, x_axis, force_binomial, var1Label=var1Label,
               var2Label=var2Label, _color=color)    

    x_points = _manual_integration(pmf, force_binomial, color=color)
    _adjust_visuals(x_points, force_binomial)

    plt.show()
