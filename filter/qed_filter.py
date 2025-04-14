from rdkit.Chem import QED

from chemtsv2.filter import Filter


class QEDFilter(Filter):
    def check(mol, conf):
        is_valid_key = False
        if 'max' in conf['QED_filter']['type']:
            is_valid_key = True
            qed_val = QED.weights_max(mol)
        if 'mean' in conf['QED_filter']['type']:
            is_valid_key = True
            qed_val = QED.weights_mean(mol)
        if 'none' in conf['QED_filter']['type']:
            is_valid_key = True
            qed_val = QED.weights_none(mol)
        if not is_valid_key:
            print("`use_QED_filter` only accepts [max, mean, none]")
            sys.exit(1)
        return qed_val > conf['QED_filter']['threshold']