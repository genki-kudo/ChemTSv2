from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from chemtsv2.filter import Filter

def acceptor_tf(mol, conf):
    add_substituent = Chem.DeleteSubstructs(mol, conf['init_mol'])
    # acceptor_num = rdMolDescriptors.CalcNumLipinskiHBA(mol)
    # acceptor_num = rdMolDescriptors.CalcNumLipinskiHBA(add_substituent)
    acceptor_num = rdMolDescriptors.CalcNumLipinskiHBA(mol) - conf['init_acceptor']
    max_min_range = conf['Dscore_parameters']['acceptor']
    if max_min_range['min'] <= acceptor_num and acceptor_num <= max_min_range['max']:
        return True
    else:
        return False

def donor_tf(mol, conf):
    add_substituent = Chem.DeleteSubstructs(mol, conf['init_mol'])
    # donor_num = rdMolDescriptors.CalcNumLipinskiHBD(mol)
    # donor_num = rdMolDescriptors.CalcNumLipinskiHBD(add_substituent)
    donor_num = rdMolDescriptors.CalcNumLipinskiHBD(mol) - conf['init_donor']
    max_min_range = conf['Dscore_parameters']['donor']
    if max_min_range['min'] <= donor_num and donor_num <= max_min_range['max']:
        return True
    else:
        return False

class Donor_Acceptor(Filter):
    def check(mol, conf):
        donor = donor_tf(mol, conf)
        acceptor = acceptor_tf(mol, conf)
        cond =  donor and acceptor 

        return cond