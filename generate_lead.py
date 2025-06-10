import sys, os, subprocess, yaml
import pandas as pd
from pathlib import Path
import rdkit
from rdkit import Chem
from IPython.core.debugger import Pdb

from ChemTSv2.chemts_mothods import Methods, logs_dir
import logging

cwd = os.path.dirname(os.path.abspath(__file__))

class Generate_Lead:
    def __init__(self, trajectory_dirs, config):
        self.trajectory_dirs = trajectory_dirs
        self.rank_output_dirs = []
        self.input_compound_files = []
        self.conf = config  
        self.out_log_file = os.path.join(self.conf['OUTPUT']['directory'], self.conf['OUTPUT']['logs_dir'], 'ChemTS.log')
        cm = Methods(self.conf)
        self.cm = cm
        self.target_dirname = self.conf['ChemTS']['target_dirname']
        self.logger = self.cm.setup_custom_logger('ChemTS', self.out_log_file)

    def run(self):
        for trajectory_dir in self.trajectory_dirs:
            self.logger.info(trajectory_dir)
            sincho_result_file = os.path.join(trajectory_dir, 'sincho_result.yaml')

            trajectory_name = trajectory_dir.split(os.sep)[-1]
            trajectory_num = trajectory_name.split('_')[-1]

            ChemTS_output_dir = os.path.join(self.conf['OUTPUT']['directory'], self.conf['ChemTS']['working_directory'])
            trajectory_output_dir = os.path.join(ChemTS_output_dir, trajectory_name)
            os.makedirs(trajectory_output_dir, exist_ok = True)
            
            with open(sincho_result_file, 'r')as f:
                sincho_results = yaml.safe_load(f)
            sincho_results = sincho_results['SINCHO_result']
            
            input_compound_file = os.path.join(trajectory_dir, 'lig_'+ trajectory_num + '.pdb')
            self.input_compound_files.append(input_compound_file)
            input_compound_smiles = Chem.MolToSmiles(Chem.MolFromPDBFile(input_compound_file))

            # 中性ならTrue,電荷ありならFalse
            self.is_neutral = self.cm.check_neutral(input_compound_smiles)
            
            for rank, sincho_result in sincho_results.items():
                self.logger.info(f"rank , {rank}")
                rank_output_dir = os.path.join(trajectory_output_dir, rank)
                os.makedirs(rank_output_dir, exist_ok = True)
                self.target_dirname = rank_output_dir
                self.rank_output_dirs.append(rank_output_dir)

                # 生やしたい分子量を取得
                estimate_add_mw = sincho_result['mw']
                weight_model_dir = self.cm.select_weight_model(input_compound_smiles, estimate_add_mw)
                self.logger.info(f"weight_model_dir , {weight_model_dir}")
                
                extend_atom = sincho_result['atom_num'].split('.')[1].split('_')[-1]

                if not self.is_neutral:
                    # 電荷ありをopenbabelで中性化する
                    # SINCHOは電荷ありでやってる
                    # 中性化→プロパティ計算→SMILES並び替え
                    # lig_000.pdb -> lig_000_org.pdbとして保持し、中性化したものをlig_000.pdbとする
                    # pdbだと上手くいかないからmol2経由する lig_000.pdb -> lig_000.mol2 -> (neutral) -> lig_000.pdb(同名だが中性化されている)
                    
                    self.logger.info('ligand has charges.')
                    # openbabelで中性化
                    self.cm.do_neutral(input_compound_file)
                    # SMILESを中性化に更新
                    input_compound_smiles = Chem.MolToSmiles(Chem.MolFromPDBFile(input_compound_file))

                # 初期SMILESの物性値を計算し、configに記載しておく
                config_add_props = self.cm.calc_property(input_compound_smiles, self.conf)

                # SMILESの並び替え(中性化→計算→並び替えの順序は保持する)
                rearrange_smi = self.cm.set_rearrange_smiles(input_compound_file, extend_atom)
                self.logger.info(f"smi , {input_compound_smiles}")

                # 不正SMILESのチェック(現状は[n]のみ)
                if not self.cm.check_error_smiles(rearrange_smi):
                    rearrange_smi = self.cm.modify_smiles(rearrange_smi)

                self.logger.info(f"rearrange_smi , {rearrange_smi}")
                
                subprocess.run(['rm', os.path.join(self.target_dirname, '_setting.yaml')])
                self.cm.make_config_file({**config_add_props, **sincho_result}, weight_model_dir, self.target_dirname)

                # 化合物生成をn回
                df_result_all = pd.DataFrame()
                for n in range(1, int(self.conf['ChemTS']['num_chemts_loops'])+1):
                    with open(self.out_log_file, 'a') as stdout_f:
                        subprocess.run(['python', '/ChemTSv2/run.py', '-c', os.path.join(self.target_dirname, '_setting.yaml'), '-t', self.target_dirname, 
                                        '--input_smiles', rearrange_smi], stdout=stdout_f, stderr=stdout_f)
                        subprocess.run(' '.join(['mv', os.path.join(self.target_dirname, 'result_C*'), 
                                                       os.path.join(self.target_dirname, 'result.csv')]), 
                                                       shell=True, stdout=stdout_f, stderr=stdout_f)
                        df_result_one_cycle = pd.read_csv(os.path.join(self.target_dirname, 'result.csv'))
                        df_result_one_cycle.insert(0, 'trial', n) 
                        df_result_all = pd.concat([df_result_all, df_result_one_cycle])
                        subprocess.run(' '.join(['cat', os.path.join(self.target_dirname, 'run.log'), 
                                                 '>>', os.path.join(self.target_dirname, 'run.log.all')]), shell=True )
                        
                # for debug df_result_all
                # df_result_all = pd.read_csv(os.path.join(self.target_dirname, 'results.csv'))
                
                # n回分を一つのファイルにし、個々のファイルは消しておく
                output_csv_path = os.path.join(self.target_dirname, 'results.csv')
                df_result_all.to_csv(output_csv_path)
                subprocess.run(['rm', os.path.join(self.target_dirname, 'result.csv'),], cwd=cwd)
                
                # 今回の生成のrewardなどをプロット
                self.cm.plot_reward(output_csv_path)

                subprocess.run(' '.join(['mv', os.path.join(self.target_dirname, '*'), rank_output_dir]), shell=True)
