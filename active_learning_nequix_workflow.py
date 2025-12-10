#!/usr/bin/env python3
"""
Active-Learning-Nequix-Workflow.py

Archivo principal (CLI) que orquesta un ciclo de Active Learning con Nequix
como calculadora ML y VASP como "oracle" DFT.

Estructura y responsabilidades:
- Cargar configuración desde config.yaml (o argumentos CLI)
- Ejecutar MD (ASE NPT) usando un modelo Nequix (uno del ensemble) o un ML rápido
- Evaluar pool de estructuras con un ensemble de Nequix (5 modelos) y computar
  medias/varianzas de energía y fuerzas
- Seleccionar top-K frames por score = E_var + F_var
- Generar inputs VASP (POSCAR/INCAR/KPOINTS/POTCAR(s)) para los frames seleccionados
- Recolectar resultados DFT (OUTCAR/vasprun.xml) y convertir a extxyz + metadata
- Invocar reentrenamiento (placeholder: llama a un script externo `train_ensemble.sh`)

NOTAS:
- Este script asume que existen scripts auxiliares (en scripts/) para entrenamiento
  y envío de jobs. No ejecuta VASP por sí mismo, apenas prepara ficheros y puede
  lanzar un script sbatch cuando se solicite.
- Contiene implementaciones tolerantes a errores para cargar checkpoints Nequix.

AUXILIARY: Este archivo es el orchestrador principal; los módulos concretos de
inferencia / entrenamiento pueden encontrarse en `scripts/`.

"""

import argparse
import yaml
import logging
from pathlib import Path
import sys
import os
import json
import shutil
import subprocess
import time

from ase import Atoms
from ase.io import read, write
import numpy as np

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('AL-Nequix')

# ----------------------------------------------------------------------------
# Helpers: load config
# ----------------------------------------------------------------------------

def load_config(path: Path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg

# ----------------------------------------------------------------------------
# Model loading helpers (Nequix)
# ----------------------------------------------------------------------------

def load_nequix_model(path: Path):
    """Intento robusto de cargar un checkpoint Nequix.
    Se prueba NequixModel.load(path) y luego fallback a torch.load(state_dict).
    Retorna un objeto 'model' o lanza RuntimeError con mensaje explicativo.
    """
    path_str = str(path)
    try:
        from nequix import NequixModel
        log.info(f'Try NequixModel.load({path_str})')
        model = NequixModel.load(path_str)
        return model
    except Exception as e1:
        log.warning(f'NequixModel.load failed: {e1}. Trying torch.load fallback...')
    try:
        import torch
        ckpt = torch.load(path_str, map_location='cpu')
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            log.info('Found model_state_dict in checkpoint; returning raw state_dict (user must instantiate model).')
            return ckpt
        # sometimes the entire model object is saved
        if hasattr(ckpt, '__dict__') or isinstance(ckpt, object):
            log.info('Checkpoint appears to be a full object; returning it directly.')
            return ckpt
        raise RuntimeError('Unknown checkpoint format in torch.load result')
    except Exception as e2:
        raise RuntimeError(f'No se pudo cargar checkpoint Nequix desde {path_str}: {e2}')


def model_to_ase_calculator(model):
    """
    Convierte el objeto `model` cargado a una ase.Calculator utilizando
    `nequixase.NequixCalculator` si está disponible. Si el objeto devuelto por
    load_nequix_model es un state_dict, este wrapper no intentará reconstruir
    la arquitectura automática — el usuario debe proporcionar una función
    externa para reconstruir el modelo completo.
    """
    try:
        from nequixase import NequixCalculator
        # Si el usuario pasó path en vez de objeto, NequixCalculator podría aceptar path
        # soportamos ambos casos
        if isinstance(model, (str, Path)):
            return NequixCalculator(model=str(model), backend='torch')
        return NequixCalculator(model=model, backend='torch')
    except Exception as e:
        log.warning(f'nequixase.NequixCalculator no disponible o fallo: {e}. Usando wrapper ASE simple.')

    # Fallback: crear un Calculator ASE mínimo que lanza error informativo.
    from ase.calculators.calculator import Calculator, all_changes

    class SimpleNequixCalculator(Calculator):
        implemented_properties = ['energy', 'forces']

        def __init__(self, model_obj, **kwargs):
            super().__init__(**kwargs)
            self.model = model_obj

        def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            raise RuntimeError('SimpleNequixCalculator: requiere implementar inferencia concreta para su checkpoint.\n' \
                               'Sugerencia: convierta el checkpoint a un objeto NequixModel o instancie NequixCalculator disponible.')

    return SimpleNequixCalculator(model)

# ----------------------------------------------------------------------------
# Ensemble prediction
# ----------------------------------------------------------------------------

def predict_ensemble_for_atoms(atoms: Atoms, model_paths, scratch_calc=None):
    """
    Dado un objeto Atoms y una lista de rutas a checkpoints (model_paths),
    devuelve dict con E_mean, E_var, F_mean, F_var.
    """
    energies = []
    forces = []
    for mp in model_paths:
        try:
            model = load_nequix_model(Path(mp))
            calc = model_to_ase_calculator(model)
            atoms.set_calculator(calc)
            E = atoms.get_potential_energy()
            F = atoms.get_forces().copy()
            energies.append(E)
            forces.append(F)
        except Exception as e:
            log.error(f'Error prediciendo con modelo {mp}: {e}')
            raise
    E = np.array(energies)
    F = np.array(forces)  # shape (n_models, natoms, 3)
    return {
        'E_mean': float(E.mean()),
        'E_var': float(E.var()),
        'F_mean': F.mean(axis=0).tolist(),
        'F_var': float(F.var(axis=0).mean())
    }

# ----------------------------------------------------------------------------
# Pool scoring and selection
# ----------------------------------------------------------------------------

def score_pool(pool_dir: Path, ensemble_paths, out_json: Path):
    pool_dir = Path(pool_dir)
    files = sorted([p for p in pool_dir.iterdir() if p.suffix in ('.xyz', '.extxyz', '.traj', '.vasp')])
    scores = {}
    for f in files:
        try:
            atoms = read(str(f))
            info = predict_ensemble_for_atoms(atoms, ensemble_paths)
            info['source'] = str(f.name)
            scores[f.name] = info
            log.info(f'Scored {f.name}: E_var={info["E_var"]:.4e} F_var={info["F_var"]:.4e}')
        except Exception as e:
            log.error(f'Failed scoring {f}: {e}')
    with open(out_json, 'w') as fh:
        json.dump(scores, fh, indent=2)
    return scores


def select_topk(scores_json: Path, pool_dir: Path, target_dir: Path, top_k: int = 10):
    with open(scores_json) as fh:
        scores = json.load(fh)
    items = []
    for name, v in scores.items():
        score = v.get('E_var', 0.0) + v.get('F_var', 0.0)
        items.append((name, score))
    items.sort(key=lambda x: x[1], reverse=True)
    selected = [it[0] for it in items[:top_k]]
    target_dir.mkdir(parents=True, exist_ok=True)
    for s in selected:
        src = pool_dir / s
        dst = target_dir / s
        try:
            shutil.copy(str(src), str(dst))
        except Exception as e:
            log.error(f'No se pudo copiar {src} -> {dst}: {e}')
    log.info(f'Selected {len(selected)} frames for DFT: {selected}')
    return selected

# ----------------------------------------------------------------------------
# Generate VASP inputs
# ----------------------------------------------------------------------------

def generate_vasp_inputs(selected_dir: Path, out_dir: Path, dft_cfg: dict):
    selected_dir = Path(selected_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in selected_dir.iterdir() if p.suffix in ('.xyz', '.extxyz', '.vasp')])
    for f in files:
        atoms = read(str(f))
        base = f.stem
        jobdir = out_dir / base
        jobdir.mkdir(exist_ok=True)
        write(str(jobdir / 'POSCAR'), atoms, format='vasp')
        # copy INCAR/KPOINTS
        incar_template = dft_cfg['vasp'].get('incar_template')
        kpoints_template = dft_cfg['vasp'].get('kpoints_template')
        potcar_dir = dft_cfg['vasp'].get('potcar_dir')
        if incar_template:
            shutil.copy(incar_template, jobdir / 'INCAR')
        if kpoints_template:
            shutil.copy(kpoints_template, jobdir / 'KPOINTS')
        # POTCAR: user may store individual POTCARs per element or a combined POTCAR
        pseudo = dft_cfg['vasp'].get('pseudo')
        if potcar_dir and pseudo:
            src = Path(potcar_dir) / pseudo
            if src.exists():
                shutil.copy(src, jobdir / 'POTCAR')
            else:
                log.warning(f'POTCAR {src} not found; user must place POTCAR manually in {jobdir}')
        # write a simple submit script per jobdir
        submit_sh = jobdir / 'run_vasp.sh'
        with open(submit_sh, 'w') as fh:
            fh.write("""#!/bin/bash\n# Simple VASP submit placeholder\nmpirun -np {nproc} vasp_std > vasp.out\n""".format(nproc=dft_cfg['vasp'].get('nproc', 16)))
        submit_sh.chmod(0o755)
    log.info(f'VASP inputs generated in {out_dir}')

# ----------------------------------------------------------------------------
# Collect VASP outputs
# ----------------------------------------------------------------------------

def parse_vasp_job(jobdir: Path):
    vasprun = jobdir / 'vasprun.xml'
    outcar = jobdir / 'OUTCAR'
    atoms = None
    meta = {'parsed': False}
    try:
        if vasprun.exists():
            atoms = read(str(vasprun), index=-1)
        elif outcar.exists():
            atoms = read(str(outcar), index=-1)
    except Exception as e:
        log.error(f'Error parsing {jobdir}: {e}')
    if atoms is not None:
        try:
            E = atoms.get_potential_energy()
            F = atoms.get_forces().tolist()
        except Exception:
            E = None
            F = None
        try:
            mag = atoms.get_magnetic_moments().tolist()
        except Exception:
            mag = None
        meta = {'energy': float(E) if E is not None else None, 'forces': F, 'magmoms': mag, 'parsed': True}
    return atoms, meta


def collect_all_results(dft_submit_dir: Path, labeled_dir: Path):
    dft_submit_dir = Path(dft_submit_dir)
    labeled_dir = Path(labeled_dir)
    labeled_dir.mkdir(parents=True, exist_ok=True)
    for jd in sorted([p for p in dft_submit_dir.iterdir() if p.is_dir()]):
        atoms, meta = parse_vasp_job(jd)
        base = jd.name
        if meta.get('parsed'):
            outpath = labeled_dir / (base + '.extxyz')
            write(str(outpath), atoms, format='extxyz')
            with open(str(outpath) + '.json', 'w') as fh:
                json.dump(meta, fh, indent=2)
            log.info(f'Collected DFT result for {base}')
        else:
            log.warning(f'No parsed result for {base}; check job {jd}')

# ----------------------------------------------------------------------------
# Retrain hook (placeholder)
# ----------------------------------------------------------------------------

def retrain_ensemble(train_script: str, dataset_dir: Path, out_models_dir: Path):
    """
    Ejecuta un script externo (bash/python) que entrena el ensemble.
    Este script es un placeholder: la lógica de entrenamiento depende del
    entorno y del toolkit de Nequix/NequIP que use el usuario.
    """
    cmd = [train_script, str(dataset_dir), str(out_models_dir)]
    log.info('Launching retrain script: ' + ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        log.error(f'Retraining failed: {e}')
        raise

# ----------------------------------------------------------------------------
# CLI: orquestador
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Active Learning orchestrator for Nequix + VASP')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--round', type=int, default=0, help='Active Learning round index')
    parser.add_argument('--run-md', action='store_true', help='Run MD to generate pool structures (uses Nequix single model)')
    parser.add_argument('--score-pool', action='store_true', help='Score pool with ensemble')
    parser.add_argument('--select', action='store_true', help='Select top_k frames for DFT')
    parser.add_argument('--gen-vasp', action='store_true', help='Generate VASP inputs for selected frames')
    parser.add_argument('--collect', action='store_true', help='Collect DFT outputs into labeled_dir')
    parser.add_argument('--retrain', action='store_true', help='Call retrain script to rebuild ensemble')
    parser.add_argument('--all', action='store_true', help='Run full cycle: score -> select -> gen -> (user runs DFT) -> collect -> retrain')
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    base_dir = Path('.')
    pool_dir = Path(cfg['paths']['pool_dir'])
    selected_dir = base_dir / cfg['paths']['dft_submit_dir'] / f'round_{args.round}' / 'selected'
    vasp_out_dir = base_dir / cfg['paths']['dft_submit_dir'] / f'round_{args.round}'
    labeled_dir = Path(cfg['paths']['labeled_dir']) / f'round_{args.round}'
    ensemble_dir = Path(cfg['ensemble']['checkpoints_dir'])
    ensure_dirs = [pool_dir, selected_dir.parent, labeled_dir]
    for d in ensure_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    # 1) (optional) Run MD: user-provided implementation or simple placeholder
    if args.run_md:
        log.info('MD run requested. This script includes only a minimal MD example; adapt to your needs.')
        # Minimal example: use first model of ensemble to run short NVT/NPT md
        try:
            from ase.md.verlet import VelocityVerlet
            from ase.md.npt import NPT
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
            from ase import units
            from ase.io.trajectory import Trajectory

            # Read base structure
            base_poscar = cfg['system'].get('base_structure')
            atoms = read(base_poscar)
            # attach calculator from first model
            first_model = load_nequix_model(Path(cfg['ensemble']['checkpoints_dir']) / cfg['ensemble']['model_names'][0])
            calc = model_to_ase_calculator(first_model)
            atoms.set_calculator(calc)

            temp = float(cfg['md']['temperature'])
            dt_fs = float(cfg['md']['timestep_fs'])
            nsteps = int(cfg['md']['nsteps'])

            MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
            dyn = VelocityVerlet(atoms, dt_fs * units.fs)
            traj = Trajectory(str(pool_dir / f'round_{args.round}.traj'), 'w', atoms)

            def write_traj(a=atoms):
                traj.write(a)

            dyn.attach(write_traj, interval=10)
            log.info(f'Starting MD: T={temp} K, steps={nsteps}, dt={dt_fs} fs')
            dyn.run(nsteps)
            traj.close()
            log.info('MD finished and trajectory saved in pool_dir')
        except Exception as e:
            log.error(f'MD step failed: {e}')
            sys.exit(1)

    # 2) Score pool
    if args.score_pool or args.all:
        scores_json = Path('pool_scores.json')
        model_paths = [str(Path(p)) for p in cfg['ensemble'].get('model_paths', [])]
        # If ensemble dir with pattern
        if not model_paths:
            # try to list files in ensemble_dir
            model_paths = sorted([str(p) for p in ensemble_dir.iterdir() if p.suffix in ('.pt', '.pth')])[:cfg['ensemble'].get('n_models', 5)]
        log.info(f'Using ensemble models: {model_paths}')
        score_pool(pool_dir, model_paths, scores_json)

    # 3) Select
    if args.select or args.all:
        top_k = int(cfg['selection'].get('top_k', 10))
        scores_json = Path('pool_scores.json')
        select_topk(scores_json, pool_dir, selected_dir, top_k=top_k)

    # 4) Generate VASP inputs
    if args.gen_vasp or args.gen_vasp or args.all:
        generate_vasp_inputs(selected_dir, vasp_out_dir, cfg)

    # 5) Collect DFT outputs (user must run DFT between gen and collect)
    if args.collect or args.all:
        collect_all_results(vasp_out_dir, labeled_dir)

    # 6) Retrain
    if args.retrain or args.all:
        train_script = cfg.get('train_script', None)
        if not train_script:
            log.error('No train_script defined in config.yaml; cannot retrain ensemble automatically.')
        else:
            retrain_ensemble(train_script, labeled_dir, Path(cfg['ensemble']['checkpoints_dir']))

    log.info('Workflow execution complete')

if __name__ == '__main__':
    main()
