import numpy as np
from scipy import stats
from scipy.stats import ttest_rel


def calculate_statistics(data_list):
    mean_val = np.mean(data_list)
    std_val = np.std(data_list, ddof=1)
    n = len(data_list)
    t_value = stats.t.ppf(0.975, df=n - 1)
    margin_error = t_value * (std_val / np.sqrt(n))
    ci_lower = mean_val - margin_error
    ci_upper = mean_val + margin_error

    return {'mean': mean_val, 'std': std_val, 'ci95_lower': ci_lower, 'ci95_upper': ci_upper,
            'values': data_list}


def calc_pvalue(our_data, test_data):
    if len(our_data) == 1:
        return np.nan
    else:
        _, pval = ttest_rel(our_data, test_data, alternative="greater")
        return pval
