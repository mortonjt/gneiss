# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from skbio import TreeNode
from skbio.stats.composition import clr, centralize

from gneiss.plugin_setup import plugin
from gneiss.plot._radial import radialplot
from gneiss.plot._heatmap import heatmap

from gneiss.regression._ols import OLSModel
from gneiss.regression._mixedlm import LMEModel
from gneiss.regression._type import (LinearRegression_g,
                                     LinearMixedEffects_g)
from q2_types.tree import Phylogeny, Rooted
from q2_composition.plugin_setup import Composition
from q2_types.feature_table import FeatureTable
from qiime2.plugin import Int, MetadataCategory

from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import row
from bokeh.models import HoverTool, BoxZoomTool, ResetTool


def _projected_prediction(model):
    """ Create projected prediction plot

    Parameters
    ----------
    model : RegressionModel

    Returns
    -------
    bokeh plot
    """
    hover = HoverTool(
            tooltips=[
                ("#SampleID", "@index"),
            ]
        )

    pred = model.predict()
    pcvar = model.percent_explained()
    pred['color'] = 'predicted'  # make predictions red
    raw = model.balances
    raw['color'] = 'raw'   # make raw values blue

    p = figure(plot_width=400, plot_height=400,
               tools=[hover, BoxZoomTool(), ResetTool()])
    raw_source = ColumnDataSource(raw)
    pred_source = ColumnDataSource(pred)

    p.circle(raw.columns[0], raw.columns[1], size=7,
             source=raw_source, fill_color='blue', legend='raw')
    p.circle(pred.columns[0], pred.columns[1], size=7,
             source=pred_source, fill_color='red', legend='predicted')

    p.title.text = 'Projected Prediction'
    p.title_location = 'above'
    p.title.align = 'center'
    p.title.text_font_size = '18pt'

    p.xaxis.axis_label = '{} ({:.2%})'.format(pcvar.index[0], pcvar.iloc[0])
    p.yaxis.axis_label = '{} ({:.2%})'.format(pcvar.index[1], pcvar.iloc[1])
    return p


def _projected_residuals(model):
    """ Create projected residual plot

    Parameters
    ----------
    model : RegressionModel

    Returns
    -------
    bokeh plot
    """
    hover = HoverTool(
            tooltips=[
                ("#SampleID", "@index"),
            ]
        )
    pcvar = model.percent_explained()
    resid = model.residuals()
    p = figure(plot_width=400, plot_height=400,
               tools=[hover, BoxZoomTool(), ResetTool()])
    resid_source = ColumnDataSource(resid)

    p.circle(resid.columns[0], resid.columns[1], size=7,
             source=resid_source, fill_color='blue', legend='residuals')

    p.title.text = 'Projected Residuals'
    p.title_location = 'above'
    p.title.align = 'center'
    p.title.text_font_size = '18pt'
    p.xaxis.axis_label = '{} ({:.2%})'.format(pcvar.index[0], pcvar.iloc[0])
    p.yaxis.axis_label = '{} ({:.2%})'.format(pcvar.index[1], pcvar.iloc[1])
    return p


def _decorate_tree(t, series):
    """ Attaches some default values on the tree for plotting.

    Parameters
    ----------
    t: skbio.TreeNode
        Input tree
    series: pd.Series
        Input pandas series

    """
    for i, n in enumerate(t.postorder()):
        n.size = 30
        if n.is_root():
            n.size = 50
        elif n.name == n.parent.children[0].name:
            n.color = '#00FF00'  # left child is green
        else:
            n.color = '#FF0000'  # right child is red

        if not n.is_tip():
            t.length = series.loc[n.name]
    return t


def _deposit_results(model, output_dir):
    """ Store all of the core regression results into a folder. """
    pred = model.predict()
    pred.to_csv(os.path.join(output_dir, 'predicted.csv'),
                header=True, index=True)
    coefficients = model.coefficients()
    coefficients.to_csv(os.path.join(output_dir, 'coefficients.csv'),
                        header=True, index=True)
    residuals = model.residuals()
    residuals.to_csv(os.path.join(output_dir, 'residuals.csv'),
                     header=True, index=True)
    predicted = model.predict()
    predicted.to_csv(os.path.join(output_dir, 'predicted.csv'),
                     header=True, index=True)
    pvalues = model.pvalues
    pvalues.to_csv(os.path.join(output_dir, 'pvalues.csv'),
                   header=True, index=True)
    balances = model.balances
    balances.to_csv(os.path.join(output_dir, 'balances.csv'),
                    header=True, index=True)


def _deposit_results_html(index_f):
    """ Create links to all of the regression results. """
    index_f.write(('<th>Coefficients</th>\n'))
    index_f.write(('<a href="coefficients.csv">'
                   'Download as CSV</a><br>\n'))
    index_f.write(('<th>Coefficient pvalues</th>\n'))
    index_f.write(('<a href="pvalues.csv">'
                   'Download as CSV</a><br>\n'))
    # TODO: Remove balances (since we will pass them in directly)
    index_f.write(('<th>Raw Balances</th>\n'))
    index_f.write(('<a href="balances.csv.csv">'
                   'Download as CSV</a><br>\n'))
    index_f.write(('<th>Predicted Proportions</th>\n'))
    index_f.write(('<a href="predicted.csv">'
                   'Download as CSV</a><br>\n'))
    index_f.write(('<th>Residuals</th>\n'))
    index_f.write(('<a href="residuals.csv">'
                   'Download as CSV</a><br>\n'))
    # TODO: Add tree

# OLS summary
def ols_summary(output_dir: str, model: OLSModel, ndim=10) -> None:
    """ Summarizes the ordinary least squares fit.

    Parameters
    ----------
    output_dir : str
        Directory where all of the regression results and
        summaries will be stored.
    model : OLSModel
        Ordinary Least Squares model that contains the model fit and the
        regression results.
    ndim : int
        The number of dimensions to summarize.
    """
    # Cross validation
    cv = model.loo()
    # Relative importance of explanatory variables
    relimp = model.lovo()
    w, h = 400, 400  # plot width and height
    # Histogram of model mean squared error from cross validation
    mse_p = figure(title="Cross Validation Mean Squared Error",
                   plot_width=w, plot_height=h)
    mse_hist, edges = np.histogram(cv.mse, density=True, bins=20)
    mse_p.quad(top=mse_hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="#FFFF00", line_color="#033649", fill_alpha=0.5,
               legend='CV Mean Squared Error')
    mse_p.ray(x=model.mse, y=0, length=h,
              angle=1.57079633, color='red',
              legend='Model Error', line_width=0.5)

    # Histogram of prediction error from cross validation
    pred_p = figure(title="Prediction Error",
                    plot_width=w, plot_height=h)
    pred_hist, edges = np.histogram(cv.pred_err, density=True, bins=20)
    pred_p.quad(top=pred_hist, bottom=0, left=edges[:-1], right=edges[1:],
                fill_color="#00FFFF", line_color="#033649", fill_alpha=0.5,
                legend='Prediction Error')
    pred_p.ray(x=model.mse, y=0, length=h,
               angle=1.57079633, color='red',
               legend='Model Error', line_width=0.5)

    cvp = row(mse_p, pred_p)

    # Explained sum of squares
    ess = pd.Series({r.model.endog_names: r.ess for r in model.results})
    # Summary object
    smry = model.summary(ndim=10)

    t = _decorate_tree(model.tree, ess)

    p1 = radialplot(t, edge_color='color', figsize=(800, 800))
    p1.title.text = 'Explained Sum of Squares'
    p1.title_location = 'above'
    p1.title.align = 'center'
    p1.title.text_font_size = '18pt'

    # 2D scatter plot for prediction on PB
    p2 = _projected_prediction(model)
    p3 = _projected_residuals(model)

    p23 = row(p2, p3)

    _deposit_results(model, output_dir)

    index_fp = os.path.join(output_dir, 'index.html')
    with open(index_fp, 'w') as index_f:
        index_f.write('<html><body>\n')
        index_f.write('<h1>Simplicial Linear Regression Summary</h1>\n')
        index_f.write(smry.as_html())
        index_f.write('<th>Relative importance</th>\n')
        index_f.write(relimp.to_html())
        _deposit_results_html(index_f)
        index_f.write('<th>Cross Validation</th>')
        cv_html = file_html(cvp, CDN, 'Cross Validation')
        index_f.write(cv_html)
        ess_tree_html = file_html(p1, CDN, 'Explained Sum of Squares')
        index_f.write(ess_tree_html)
        reg_smry_html = file_html(p23, CDN, 'Prediction and Residual plot')
        index_f.write(reg_smry_html)
        index_f.write('</body></html>\n')


plugin.visualizers.register_function(
    function=ols_summary,
    inputs={'model': LinearRegression_g},
    parameters={'ndim': Int},
    input_descriptions={
        'model': 'The fitted simplicial ordinary least squares model.'
    },
    parameter_descriptions={
        'ndim': 'Number of dimensions to summarize.'},
    name='Simplicial Linear Regression Summary plots.',
    description=("Visualize the summary statistics of simplicial "
                 "linear regression plot. This includes the "
                 "explained sum of squares, coefficients, "
                 "coefficient pvalues, coefficient of determination, "
                 "predicted fit, and residuals")
)


# LME summary
def lme_summary(output_dir: str, model: LMEModel, ndim=10) -> None:
    """ Summarizes the ordinary linear mixed effects model.

    Parameters
    ----------
    output_dir : str
        Directory where all of the regression results and
        summaries will be stored.
    model : LMEModel
        Linear Mixed Effects model that contains the model fit and the
        regression results.
    ndim : int
        The number of dimensions to summarize.
    """
    # log likelihood
    loglike = pd.Series({r.model.endog_names: r.model.loglike(r.params)
                         for r in model.results})

    # Summary object
    smry = model.summary(ndim=10)

    t = _decorate_tree(model.tree, -loglike)
    p1 = radialplot(t, edge_color='color', figsize=(800, 800))
    p1.title.text = 'Loglikelihood of submodels'
    p1.title_location = 'above'
    p1.title.align = 'center'
    p1.title.text_font_size = '18pt'

    # 2D scatter plot for prediction on PB
    p2 = _projected_prediction(model)
    p3 = _projected_residuals(model)

    p23 = row(p2, p3)

    # Deposit all regression results
    _deposit_results(model, output_dir)

    index_fp = os.path.join(output_dir, 'index.html')
    with open(index_fp, 'w') as index_f:
        index_f.write('<html><body>\n')
        index_f.write('<h1>Simplicial Linear Mixed Effects Summary</h1>\n')
        index_f.write(smry.as_html())
        _deposit_results_html(index_f)
        ess_tree_html = file_html(p1, CDN, 'Loglikelihood')
        index_f.write(ess_tree_html)
        reg_smry_html = file_html(p23, CDN, 'Prediction and Residual plot')
        index_f.write(reg_smry_html)
        index_f.write('</body></html>\n')


plugin.visualizers.register_function(
    function=lme_summary,
    inputs={'model': LinearMixedEffects_g},
    parameters={'ndim': Int},
    input_descriptions={
        'model': 'The fitted simplicial ordinary least squares model.'
    },
    parameter_descriptions={
        'ndim': 'Number of dimensions to summarize.'},
    name='Simplicial Linear Mixed Effects Summary plots.',
    description=("Visualize the summary statistics of a simplicial "
                 "linear mixed effectsmodel. This includes the "
                 "loglikhood estimates, coefficients, "
                 "coefficient pvalues, coefficient of determination, "
                 "predicted fit and residuals")
)


# Heatmap
def dendrogram_heatmap(output_dir: str, table: pd.DataFrame,
                       tree: TreeNode, metadata: MetadataCategory,
                       ndim=10):

    nodes = [n.name for n in tree.levelorder()]
    nlen = min(ndim, len(nodes))
    highlights = pd.DataFrame([['#00FF00', '#FF0000']] * nlen,
                              index=nodes[:nlen])

    mat = pd.DataFrame(clr(centralize(table)),
                       index=table.index,
                       columns=table.columns)

    # TODO: There are a few hard-coded constants here
    # will need to have some adaptive defaults set in the future
    fig = heatmap(mat, tree, metadata.to_series(), highlights,
                  highlight_width=0.01, figsize=(12, 8))
    fig.savefig(os.path.join(output_dir, 'heatmap.svg'))

    index_fp = os.path.join(output_dir, 'index.html')
    with open(index_fp, 'w') as index_f:
        index_f.write('<html><body>\n')
        index_f.write('<h1>Dendrogram heatmap</h1>\n')
        index_f.write('<img src="heatmap.svg" alt="heatmap">')
        index_f.write('</body></html>\n')


plugin.visualizers.register_function(
    function=dendrogram_heatmap,
    inputs={'table': FeatureTable[Composition],
            'tree': Phylogeny[Rooted]},
    parameters={'metadata': MetadataCategory, 'ndim': Int},
    input_descriptions={
        'table': ('The feature table that will be plotted as a heatmap. '
                  'This table is assumed to have strictly positive values.'),
        'tree': ('A hierarchy of feature identifiers where each tip'
                 'corresponds to the feature identifiers in the table. '
                 'This tree can contain tip ids that are not present in '
                 'the table, but all feature ids in the table must be '
                 'present in this tree.')},
    parameter_descriptions={
        'metadata': ('Metadata to group the samples. '),
        'ndim': 'Number of dimensions to highlight.'},
    name='Dendrogram heatmap.',
    description=("Visualize the feature tables as a heatmap. "
                 "with samples sorted along a specified metadata category "
                 "and features clustered together specified by the tree."
                 "In addition, the heatmap values are clr transformed.")
)
