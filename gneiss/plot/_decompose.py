# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from gneiss.util import NUMERATOR, DENOMINATOR
from sklearn.mixture import GaussianMixture
from scipy.stats import norm


def balance_boxplot(balance_name, data, num_color='#FFFFFF',
                    denom_color='#FFFFFF',
                    xlabel="", ylabel="", linewidth=1,
                    ax=None, **kwargs):
    """ Plots a boxplot for a given balance and the associated metadata.

    Parameters
    ----------
    x, y, hue: str
        Variable names to be passed into the seaborn plots for plotting.
    balance_name : str
        Name of balance to plot.
    data : pd.DataFrame
        Merged dataframe of balances and metadata.
    num_color : str
        Hex for background colors of values above zero.
    denom_color : str
        Hex for background colors of values below zero.
    xlabel : str
        x-axis label.
    ylabel : str
        y-axis label.
    linewidth : str
        Width of the grid lines.
    ax : matplotlib axes object
        Axes object to render boxplots in.
    **kwargs : dict
        Values to pass in to customize seaborn boxplot.

    Returns
    -------
    a : matplotlib axes object
        Matplotlit axes object with rendered boxplots.

    See Also
    --------
    seaborn.boxplot
    """
    import seaborn as sns
    if ax is None:
        f, ax = plt.subplots()

    # the number 20 is pretty arbitrary - we are just
    # resizing to make sure that there is separation between the
    # edges of the plot, and the boxplot
    pad = (data[balance_name].max() - data[balance_name].min()) / 20
    ax.axvspan(data[balance_name].min() - pad, 0,
               facecolor=num_color, zorder=0)
    ax.axvspan(0, data[balance_name].max() + pad,
               facecolor=denom_color, zorder=0)

    if 'hue' in kwargs.keys():
        hue = kwargs['hue']
        num_groups = len(data[hue].value_counts())
    else:
        num_groups = 1
    a = sns.boxplot(ax=ax, x=balance_name, data=data, **kwargs)
    a.minorticks_on()
    minorLocator = matplotlib.ticker.AutoMinorLocator(num_groups)
    a.get_yaxis().set_minor_locator(minorLocator)
    a.grid(axis='y', which='minor', color='k', linestyle=':', linewidth=1)
    a.set_xlim([data[balance_name].min() - pad,
                data[balance_name].max() + pad])
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    return a


def balance_barplots(tree, balance_name, header, feature_metadata,
                     ndim=5, num_color="#0000FF", denom_color="#0000FF",
                     xlabel="", ylabel="",
                     axes=(None, None)):
    """ Plots barplots of counts of features found in the balance.

    Parameters
    ----------
    tree : skbio.TreeNode
        Reference tree for balances.
    balance_name : str
        Name of balance to plot.
    header : str
        Header name for the feature metadata column to summarize
    feature_metadata : pd.DataFrame
        Contains information about the features.
    ndim : int
        Number of bars to display at a given time (default=5)
    num_color : str
        Hex for background colors of values above zero.
    denom_color : str
        Hex for background colors of values below zero.
    xlabel : str
        x-axis label.
    ylabel : str
        y-axis label.
    axes : tuple of matplotlib axes objects
        Specifies where the barplots should be rendered.

    Returns
    -------
    ax_num : matplotlib axes object
        Barplot of the features in the numerator of the balance.
    ax_denom : matplotlib axes object
        Barplot of the features in the denominator of the balance.
    """
    import seaborn as sns
    if axes[0] is None or axes[1] is None:
        f, (ax_num, ax_denom) = plt.subplots(2)
    else:
        ax_num, ax_denom = axes[0], axes[1]
    st = tree.find(balance_name)
    num_clade = st.children[NUMERATOR]
    denom_clade = st.children[DENOMINATOR]
    if num_clade.is_tip():
        num_ = pd.DataFrame(
            [[feature_metadata.loc[num_clade.name, header], 1]],
            columns=['index', header],
            index=[header])
    else:
        num = feature_metadata.loc[list(num_clade.subset())]
        num_ = num[header].value_counts().head(ndim).reset_index()

    if denom_clade.is_tip():
        denom_ = pd.DataFrame(
            [[feature_metadata.loc[denom_clade.name, header], 1]],
            columns=['index', header],
            index=[header])
    else:
        denom = feature_metadata.loc[list(denom_clade.subset())]
        denom_ = denom[header].value_counts().head(ndim).reset_index()

    ax_denom = sns.barplot(y='index', x=header, data=denom_, ax=ax_denom,
                           color=denom_color)
    ax_denom.set_ylabel(ylabel)
    ax_denom.set_xlabel(xlabel)
    ax_denom.set_xlim([0,  max([num_.max().values[1],
                                denom_.max().values[1]])])

    ax_num = sns.barplot(y='index', x=header, data=num_, ax=ax_num,
                         color=num_color)
    ax_num.set_ylabel(ylabel)
    ax_num.set_xlabel(xlabel)
    ax_num.set_xlim([0,  max([num_.max().values[1],
                              denom_.max().values[1]])])
    return ax_num, ax_denom

def balance_histogram(balance, metadata, colors=None, hist_kwargs={}, ax=None):
    """ Plots histogram of balances values over all sample classes.

    Parameters
    ----------
    balance : pd.Series
       Vector of balance values.
    metadata : pd.Series
       Metadata categories.
    colors : pd.Series
       Colors to plot for each metadata category.
    hist_kwargs : dict
       Arguments to pass into seaborn.distplot
    ax : matplotlib.pyplot.axes
        Axes object to plot histogram on, optional

    Returns
    -------
    ax : matplotlib.pyplot.axes
    """
    cats = metadata.value_counts().index
    if colors is None:
        colors = pd.Series(sns.color_palette('hls', len(cats)),
                           index=cats)
    if ax is None:
        f, ax = plt.subplots()

    for c in cats:
        sns.distplot(balance[metadata==c],
                     label=c, color=colors[c], ax=ax, **hist_kwargs)
        ax.legend()
        ax.set_ylabel('Proportion of Samples')
        ax.set_xlabel('Balance')
    return ax


def mixture_plot(spectrum, intervals=100,
                 numerator_color='r', filter_color='b', denominator_color='g',
                 fit_kwargs={}, hist_kwargs={}, ax=None):
    """ Plots Gaussian Mixtures on top of histogram.

    Parameters
    ----------
    spectrum : array_like
        Vector of values to plot for histogram
    numerator_color : str
        Color to plot numerator distribution.
    filter_color :
        Color to plot filtered distribution.
    denominator_color :
        Color to plot denominator distribution.
    fit_kwargs : dict
        Parameters to pass into the Gaussian Mixture Model.
    hist_kwargs : dict
        Parameters to pass into seaborn.distplot
    ax : matplotlib.pyplot.axes
        Axes object to plot histogram on, optional

    Returns
    -------
    ax : matplotlib.pyplot.axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    if 'n_components' not in fit_kwargs.keys():
        fit_kwargs['n_components'] = 3
    if 'norm_hist' not in fit_kwargs.keys():
        fit_kwargs['norm_hist'] = True
    if 'kde' not in fit_kwargs.keys():
        fit_kwargs['kde'] = False
    x = np.array(spectrum).reshape(1, -1).T
    mixture = GaussianMixture(**fit_kwargs)
    mixture.fit(X=x)

    m = np.ravel(mixture.means_)
    s = np.ravel(np.sqrt(mixture.covariances_))
    w = np.ravel(mixture.weights_)

    x = np.linspace(spectrum.min(), spectrum.max(), intervals)

    sns.distplot(spectrum, ax=ax, **hist_kwargs)
    ax.plot(x, w[0]*norm.pdf(x, m[0], s[0]), numerator_color)
    ax.plot(x, w[1]*norm.pdf(x, m[1], s[1]), filter_color)
    ax.plot(x, w[2]*norm.pdf(x, m[2], s[2]), denominator_color)
    return ax
