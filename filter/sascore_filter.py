import sys
import os

cwd = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(cwd,"../data/"))
import data.sascorer as sascorer

from chemtsv2.filter import Filter


class SascoreFilter(Filter):
    def check(mol, conf):
        return conf['sascore_filter']['threshold'] > sascorer.calculateScore(mol)
