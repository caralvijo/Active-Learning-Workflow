# --------------------------------------------------------------
# File: run_active_learning_nequix.py
# --------------------------------------------------------------
"""
Script orquestador. Flujo simplificado:
1) Generar pool (externo) -> data/pool_candidates
2) Score pool con ensemble -> pool_scores.json
3) Seleccionar top_k -> data/dft_submit_selected
4) Generar inputs VASP para cada selecionado
5) (Usuario envía jobs a cluster)
6) Tras completarse DFT: collect_vasp_outputs.py -> data/dft_labels
7) Reentrenar modelos / actualizar ensemble
"""
import yaml, os
from utils import ensure_dirs
from predict_uncertainty_nequix import score_pool
from select_worst_frames import select_topk_from_scores
from generate_vasp_inputs import generate_vasp_single_points
from collect_vasp_outputs import collect_all_results




def main(config_path='config.yaml'):
with open(config_path) as f:
cfg = yaml.safe_load(f)
ensure_dirs(cfg['paths']['pool_dir'], cfg['paths']['dft_submit_dir'], cfg['paths']['labeled_dir'], cfg['paths']['logs'])
# 1) Score pool
scores = score_pool(cfg['paths']['pool_dir'], cfg['ensemble']['checkpoints_dir'], out_json='pool_scores.json')
# 2) Select
selected = select_topk_from_scores('pool_scores.json', cfg['selection']['top_k'], cfg['paths']['pool_dir'], cfg['paths']['dft_submit_dir'])
print('Selected for DFT:', selected)
# 3) Generate VASP inputs
generate_vasp_single_points(cfg['paths']['dft_submit_dir'], cfg['paths']['dft_submit_dir'], cfg['dft'])
print('VASP inputs generated in', cfg['paths']['dft_submit_dir'])
# Step 4: user submits DFT jobs
# After DFT done, user calls collect_all_results


if __name__ == '__main__':
main()
