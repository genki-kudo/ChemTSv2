import subprocess
import sys
import copy
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import yaml
import rdkit
from rdkit import Chem

import ChemTSv2.chemts_methods as cm

CHEMTS_CURRENT_DIR = Path(__file__).resolve().parent

def run_chemts_trial(rearrange_smi, working_dir, setting_file_name, chemts_current_dir):
    stdout_log = working_dir / 'stdout.log'
    with open(stdout_log, 'w') as stdout_f:
        try:
            cmd = [ sys.executable, 'run.py', '-c', str(working_dir / setting_file_name), '-t', str(working_dir), '--input_smiles', rearrange_smi ]
            subprocess.run(cmd, cwd=str(chemts_current_dir), stdout=stdout_f, stderr=stdout_f, check=True)
            return True, None
        except subprocess.CalledProcessError as e:
            return False, f"Process returned non-zero exit code: {e.returncode}"
        except Exception as e:
            # 起動失敗やOSエラーなどをキャッチして戻り値として返す
            return False, str(e)

class Generate_Lead:
    def __init__(self, chemts_config, output_dir: str, logger):
        self.input_compound_files = []
        self.base_chemts_config = copy.deepcopy(chemts_config)
        self.out_log_file = logger
        self.chemts_output_dir = Path(output_dir)
        self.num_chemts_loops = int(chemts_config['num_chemts_loops'])
        self.num_parallel = int(chemts_config.get('num_parallel', 1))
        self.logger = logger

    def run(self, trajectory_dirs: List[Path]) -> List[Path]:
        all_rank_output_dirs = []
        all_tasks = []
        
        # ChemTSの引数を準備
        for trajectory_dir in trajectory_dirs:
            self.logger.info(str(trajectory_dir))
            sincho_result_file = trajectory_dir / 'sincho_result.yaml'
            with open(sincho_result_file, 'r')as f:
                sincho_results = yaml.safe_load(f)
            sincho_results = sincho_results['SINCHO_result']

            trajectory_name = trajectory_dir.name
            trajectory_num = trajectory_name.split('_')[-1]

            trajectory_output_dir = self.chemts_output_dir / trajectory_name
            trajectory_output_dir.mkdir(parents=True, exist_ok = True)
            
            
            input_compound_file = trajectory_dir / f'lig_{trajectory_num}.pdb'
            self.input_compound_files.append(input_compound_file)
            input_compound_smiles = Chem.MolToSmiles(Chem.MolFromPDBFile(str(input_compound_file)))

            # 中性ならTrue,電荷ありならFalse
            is_neutral = cm.check_neutral(input_compound_smiles)
            if not is_neutral:
                # 電荷ありをopenbabelで中性化する
                # SINCHOは電荷ありでやってる
                # 中性化→プロパティ計算→SMILES並び替え
                # lig_000.pdb -> lig_000_org.pdbとして保持し、中性化したものをlig_000.pdbとする
                # pdbだと上手くいかないからmol2経由する lig_000.pdb -> lig_000.mol2 -> (neutral) -> lig_000.pdb(同名だが中性化されている)
                
                self.logger.info('ligand has charges.')
                try:
                    # openbabelで中性化
                    cm.do_neutral(str(input_compound_file), self.logger)
                    # SMILESを中性化に更新
                    input_compound_smiles = Chem.MolToSmiles(Chem.MolFromPDBFile(str(input_compound_file)))
                except Exception as e:
                    self.logger.error(f"Neutralization failed for {trajectory_dir}: {e}")
                    continue

            for rank, sincho_result in sincho_results.items():
                self.logger.info(f"rank , {rank}")
                rank_output_dir = trajectory_output_dir / rank
                rank_output_dir.mkdir(parents=True, exist_ok = True)
                all_rank_output_dirs.append(rank_output_dir)

                # 生やしたい分子量を取得
                estimate_add_mw = sincho_result['mw']
                weight_model_dir = cm.select_weight_model(input_compound_smiles, estimate_add_mw)
                self.logger.info(f"weight_model_dir , {weight_model_dir}")
                
                try:
                    # 初期SMILESの物性値を計算し、configに記載しておく
                    ts_config = cm.ChemTSConfig(self.base_chemts_config)
                    ts_config.update_molecule_properties(input_compound_smiles)

                    # SMILESの並び替え(中性化→計算→並び替えの順序は保持する)
                    extend_atom = sincho_result['atom_num'].split('.')[1].split('_')[-1]
                    rearrange_smi = cm.set_rearrange_smiles(str(input_compound_file), extend_atom, logger = self.logger)
                    self.logger.info(f"smi , {input_compound_smiles}")

                    # 不正SMILESのチェック(現状は[n]のみ)
                    if not cm.check_error_smiles(rearrange_smi):
                        rearrange_smi = cm.modify_smiles(rearrange_smi, str(input_compound_file.parent), logger = self.logger)
                except Exception as e:
                    self.logger.error(f"Failed to process SMILES for rank {rank} in {trajectory_dir}: {e}")
                    continue

                self.logger.info(f"rearrange_smi , {rearrange_smi}")

                # 化合物生成をn回（タスクの準備のみ）
                for n in range(1, self.num_chemts_loops+1):
                    working_dir = rank_output_dir / 'working' / f'trial_{n}'
                    working_dir.mkdir(parents=True, exist_ok=True)
                    setting_file_name = '_setting.yaml'
                    
                    # 試行ごとの設定更新と保存
                    ts_config_trial = copy.deepcopy(ts_config)
                    ts_config_trial.update_trial_settings(sincho_result, weight_model_dir, working_dir, logger=self.logger)
                    ts_config_trial.save(working_dir / setting_file_name)

                    # タスクパラメータの保存
                    task_params = (n, rearrange_smi, working_dir.resolve(), setting_file_name, CHEMTS_CURRENT_DIR)
                    all_tasks.append(task_params)
                
        # ChemTSを並列実行
        total_tasks = len(all_tasks)
        completed_count = 0
        self.logger.info(f"Starting parallel execution of {total_tasks} tasks with {self.num_parallel} workers.")
        with ProcessPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = []
            for task_param in all_tasks:
                trial_n, rearrange_smi, working_dir, setting_file_name, chemts_current_dir = task_param
                future = executor.submit(run_chemts_trial, rearrange_smi, working_dir, setting_file_name, chemts_current_dir)
                futures.append((future, trial_n, working_dir))

            for future, trial_n, working_dir in futures:
                try:
                    chemts_success, error_info = future.result()
                    if not chemts_success:
                        self.logger.error(f"ChemTS execution failed in trial {trial_n} at {working_dir}: {error_info}")
                except Exception as e:
                    # プロセスの異常死（セグフォ、OOM等）が発生した場合はfuture.result()が例外を投げる
                    self.logger.error(f"ChemTS process crashed in trial {trial_n} at {working_dir}: {e}")
                completed_count += 1
                self.logger.info(f"Progress: {completed_count}/{total_tasks} trials finished.")
        self.logger.info("All parallel tasks finished.")
        
        # 集計
        for rank_output_dir in all_rank_output_dirs:
            rank_id = rank_output_dir.name
            traj_name = rank_output_dir.parent.name
            success_trials = []
            
            # 試行ディレクトリを探索して成功したものを特定
            for n in range(1, self.num_chemts_loops + 1):
                wd = rank_output_dir / 'working' / f'trial_{n}'
                if list(wd.glob('result_C*')): # 結果ファイルが存在すれば成功とみなす
                    success_trials.append((n, wd))
            
            if not success_trials:
                self.logger.error(f"All trials failed for {rank_id} in {traj_name}. Skipping aggregation.")
                continue

            # ログの集約
            with open(rank_output_dir / 'run.log.all', 'w') as f_out:
                for _, working_dir in success_trials:
                    run_log_path = working_dir / 'run.log'
                    if run_log_path.exists():
                        with open(run_log_path, 'r') as f_in:
                            f_out.write(f_in.read())

            # 結果CSVの集約
            df_result_list = []
            for trial_n, working_dir in success_trials:
                matched_files = list(working_dir.glob('result_C*'))
                if len(matched_files) == 1:
                    df = pd.read_csv(str(matched_files[0]))
                    df.insert(0, 'trial', trial_n)
                    df_result_list.append(df)
                elif len(matched_files) > 1:
                    raise RuntimeError(f"Multiple result files found in {working_dir}: {matched_files}. Expected only one.")
                else:
                    self.logger.error(f"No result file found matching 'result_C*' in {working_dir}")
            
            if df_result_list:
                df_result_all = pd.concat(df_result_list, ignore_index=True)
                # n回分を一つのファイルに集約する
                output_csv_path = rank_output_dir / 'results.csv'
                df_result_all.to_csv(str(output_csv_path))
                
                # 今回の生成のrewardなどをプロット
                cm.plot_reward(str(output_csv_path))
            else:
                self.logger.error(f"No valid results were collected for {rank_id} in {traj_name}. 'results.csv' will not be created.")

        return all_rank_output_dirs