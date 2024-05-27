from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np
from chemtsv2.misc.scaler import gauss, minmax, max_gauss, min_gauss, rectangular, trapezoid
from chemtsv2.reward import Reward

from IPython.core.debugger import Pdb

def scale_objective_value(params, value):
    scaling = params["type"]
    if params['type'] == 'gauss':
        return gauss(value, params["alpha"], params["mu"], params["sigma"])
    elif scaling == "max_gauss":
        return max_gauss(value, params["alpha"], params["mu"], params["sigma"])
    elif scaling == "min_gauss":
        return min_gauss(value, params["alpha"], params["mu"], params["sigma"])
    elif scaling == "minmax":
        return minmax(value, params["min"], params["max"])
    elif scaling == "rectangular":
        return rectangular(value, params["min"], params["max"])
    elif scaling == "identity":
        return value
    elif scaling == 'trapezoid':
        return trapezoid(value, params["top_min"], params["top_max"], params["bottom_min"], params["bottom_max"])
    else:
        raise ValueError("Set the scaling function from one of 'max_gauss', 'min_gauss', 'minimax', rectangular, or 'identity'")

class MW_LogP_reward(Reward):
    def get_objective_functions(conf):
        def Add_Substituent_MW(mol):
            # mw  = Descriptors.ExactMolWt(mol)
            
            # 部分削除版(Hで塞いでいる？) 
            # add_substituent = Chem.DeleteSubstructs(mol, conf['init_mol'])
            # add_substituent_mw  = Descriptors.ExactMolWt(add_substituent)

            # return mw - conf['init_mw']
            return Descriptors.ExactMolWt(mol) - conf['init_mw']

        def Add_Substituent_LogP(mol):
            # add_substituent = Chem.DeleteSubstructs(mol, conf['init_mol'])
            # return Descriptors.MolLogP(mol)
            return Descriptors.MolLogP(mol) - conf['init_logP']

        return [Add_Substituent_MW, Add_Substituent_LogP]

    def calc_reward_from_objective_values(values, conf):
        if None in values:
            return -1

        dscore_params = conf["Dscore_parameters"]
        objectives = ['MW', 'LogP']
        # objectives = [f.__name__ for f in MW_LogP_reward.get_objective_functions(conf)]

        scaled_values = []
        weights = []
        for objective, value in zip(objectives, values):
            scaled_values.append(scale_objective_value(dscore_params[objective], value))
            weights.append(dscore_params[objective]["weight"])

        multiplication_value = 1
        for v, w in zip(scaled_values, weights):
            multiplication_value *= v**w
        dscore = multiplication_value ** (1/sum(weights))

        return dscore
