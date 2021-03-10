import random 
from sklearn.utils.random import sample_without_replacement as sampler
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def sub_sampling(table, sampling_depth = 50):
    sub_sampled_table = np.zeros(table.shape)
    for idx in range(table.shape[0]):
        pool = []
        for ISM_idx, count in enumerate(table[idx].tolist()):
            pool.extend([ISM_idx] * int(count))
        sample_idxes = sampler(len(pool), sampling_depth, method='auto')
        sub_sampled_dict = Counter(np.array(pool)[sample_idxes])
        for ISM_idx in sub_sampled_dict:
            sub_sampled_table[idx, ISM_idx] = sub_sampled_dict[ISM_idx]
    return sub_sampled_table

def bray_curtis_distance(table, sample1_id, sample2_id):
    numerator = 0
    denominator = 0
    sample1_counts = table[sample1_id]
    sample2_counts = table[sample2_id]
    for sample1_count, sample2_count in zip(sample1_counts, sample2_counts):
        numerator += abs(sample1_count - sample2_count)
        denominator += sample1_count + sample2_count
    return numerator / denominator


def ism_pca(ism_df, percentile=None, sampling_depth=None, is_return_plot=True):
    # assert((percentile == None) ^ (sampling_depth == None))
    # if percentile != None: assert(percentile > 0)
    
    sampling_depth = 0
    
    country_vs_continent = dict(list(set(zip(ism_df['country/region'], ism_df['continent']))))
    
    # get sorted list of country vs sample count
    country_vs_sample_count = list(ism_df['country/region'].value_counts().items())
    
    # if percentile != None:
    #     sampling_depth = country_vs_sample_count[int(percentile * len(country_vs_sample_count))][1]
    # print(sampling_depth, sum([v for c,v in country_vs_sample_count]))
    
    # filter only countries with enough sample > sampling_depth
    filtered_countries = [c for c, v in list(ism_df['country/region'].value_counts().items()) if v >= sampling_depth]
    filt_country_vs_continent = dict([(country, cont) for country, cont in country_vs_continent.items() if country in filtered_countries])

    # Get all unique isms from this list
    ism_set = set([])
    for country in filt_country_vs_continent.keys():
        ism_set.update(ism_df.loc[ism_df['country/region'] == country, 'ISM'])
        
    # map each unique ism to a number
    ism_to_idx = dict(zip(ism_set, list(range(len(ism_set)))))
    
    # create a table of each country's ism count: 
    # each row is a country and each column represent a count of an unique ism seq
    country_vs_ism = np.zeros((len(filt_country_vs_continent), len(ism_set)))
    for c_idx, country in enumerate(filt_country_vs_continent):
        for ism, count in ism_df.loc[ism_df['country/region'] == country, 'ISM'].value_counts().items():
            ism_idx = ism_to_idx[ism]
            country_vs_ism[c_idx, ism_idx] += count
            
    # country_vs_ism_subsampled = sub_sampling(country_vs_ism, sampling_depth = sampling_depth)
    
    # d_bray_curtis = np.zeros((len(filt_country_vs_continent), len(filt_country_vs_continent)))
    # for i in range(d_bray_curtis.shape[0]):
    #     for j in range(d_bray_curtis.shape[1]):
    #         d_bray_curtis[i, j] = bray_curtis_distance(country_vs_ism_subsampled, i, j)
    
    softmax_country_vs_ism = softmax(country_vs_ism, axis=1)
    std_country_vs_ism = (country_vs_ism - np.mean(country_vs_ism, axis=1).reshape((-1,1))) / np.std(country_vs_ism, axis=1).reshape((-1,1))
    
    pca = PCA(n_components=2)
    # X_2d = pca.fit_transform(d_bray_curtis)
    X_2d = pca.fit_transform(std_country_vs_ism)
    # X_2d = pca.fit_transform(softmax_country_vs_ism)
    
    X_2d *= 100
    
    if not is_return_plot:
        return X_2d
    
    # map continent to unique color
    color_list = ['red', 'orange', 'green', 'blue', 'brown', 'black']
    continent_vs_color = dict(zip(ism_df['continent'].unique(), color_list))
    
    DPI = 150
    fig = plt.figure(figsize=(20, 15), dpi=DPI)   
    n_row = X_2d.shape[0]
    seen = set([])
    for i, country in enumerate(filt_country_vs_continent):
        continent = filt_country_vs_continent[country]
        if continent in seen:
            plt.plot(X_2d[i, 0], X_2d[i, 1], 'o', color=continent_vs_color[continent])
            
        else:
            plt.plot(X_2d[i, 0], X_2d[i, 1], 'o', color=continent_vs_color[continent], label=continent)
            seen.add(continent)
        plt.text(X_2d[i, 0] + np.random.rand()*0.1, X_2d[i, 1] + np.random.rand()*0.1, country, size=8)
    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    return X_2d, fig

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p
