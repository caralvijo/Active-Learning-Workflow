# --------------------------------------------------------------
ckpts = sorted([os.path.join(ensemble_ckpts_dir,f) for f in os.listdir(ensemble_ckpts_dir)])
energies = []
forces = []
for ck in ckpts:
model = load_nequix_model_(implementado)(ck)
calc = model_to_ase_calculator(model)
atoms.set_calculator(calc)
E = atoms.get_potential_energy()
F = atoms.get_forces().copy()
energies.append(E)
forces.append(F)
E = np.array(energies)
F = np.array(forces) # shape (n_models, natoms, 3)
return {
'E_mean': float(E.mean()),
'E_var': float(E.var()),
'F_mean': F.mean(axis=0).tolist(),
'F_var': float(F.var(axis=0).mean())
}




def model_to_ase_calculator(model):
"""Convierte un objeto de modelo Nequix a un ase.Calculator.
Implementa según cómo cargues Nequix (puede que ya exista NequixCalculator).
"""
# Ejemplo conceptual:
# return NequixCalculator(model=model, backend='torch')
try:
from nequixase import NequixCalculator
return NequixCalculator(model=model, backend="torch")
except Exception:
# Alternativa: si no existe nequixase, construye un wrapper ASE simple
try:
from ase.calculators.calculator import Calculator, all_changes
class SimpleNequixCalculator(Calculator):
implemented_properties = ['energy','forces']
def __init__(self, model, **kwargs):
super().__init__(**kwargs)
self.model = model
def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
super().calculate(atoms, properties, system_changes)
# El usuario debe implementar la inferencia concreta de su modelo aquí
raise RuntimeError('Necesario implementar inferencia de Nequix en SimpleNequixCalculator')
return SimpleNequixCalculator(model=model)
except Exception as e:
raise RuntimeError(f"No se pudo crear un ASE Calculator para el modelo: {e}")


# Si quieres, añade una función para correr sobre todo el pool y guardar resultados


def score_pool(pool_dir, ensemble_dir, out_json='pool_scores.json'):
scores = {}
files = sorted([f for f in os.listdir(pool_dir) if f.endswith('.xyz') or f.endswith('.extxyz') or f.endswith('.traj')])
for f in files:
path = os.path.join(pool_dir, f)
atoms = read(path)
info = predict_ensemble_for_atoms(atoms, ensemble_dir)
info['source_file'] = f
info['timestamp'] = now_iso()
scores[f] = info
with open(out_json, 'w') as fh:
json.dump(scores, fh, indent=2)
return scores
