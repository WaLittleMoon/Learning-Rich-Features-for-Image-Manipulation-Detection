"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from lib.datasets.pascal_voc import pascal_voc
from lib.datasets.gene_pascal_voc import gene_pascal_voc
import numpy as np

# Set up voc_<year>_<split>
for year in ['2007']:
    for split in ['trainval']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year)) # pascal_voc('trainval','2007')

# dataset newly generated 
for year in ['2007']:
    for split in ['trainval']:
        name = 'gene_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: gene_pascal_voc(split, year)) # gene_pascal_voc('trainval','2007')

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
