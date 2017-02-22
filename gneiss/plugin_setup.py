# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import importlib
from gneiss import __version__

try:
    import qiime2.plugin
    import qiime2.sdk
except ImportError:
    raise ImportWarning('Qiime2 not installed.')


plugin = qiime2.plugin.Plugin(
    name='gneiss',
    version=__version__,
    website='https://biocore.github.io/gneiss/',
    package='gneiss',
    citation_text=('Morton JT, Sanders J, Quinn RA, McDonald D, Gonzalez A, '
                   'Vázquez-Baeza Y, Navas-Molina JA, Song SJ, Metcalf JL, '
                   'Hyde ER, Lladser M, Dorrestein PC, Knight R. 2017. '
                   'Balance trees reveal microbial niche differentiation. '
                   'mSystems 2:e00162-16. '
                   'https://doi.org/10.1128/mSystems.00162-16.'))

importlib.import_module('gneiss.regression')
