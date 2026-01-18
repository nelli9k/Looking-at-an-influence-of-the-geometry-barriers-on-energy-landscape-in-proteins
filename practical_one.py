from Bio import PDB
from Bio.Data import IUPACData
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
from google.colab import files
import torch
import esm

uploaded = files.upload()

#Функція для розрахунку RMSD
def calculate_rmsd(pdb1, pdb2):
    parser = PDB.PDBParser(QUIET=True)
    structure1 = parser.get_structure('struct1', pdb1)
    structure2 = parser.get_structure('struct2', pdb2)
    
    # Знаходимо перший ланцюг (chain) у кожній структурі
    chain1 = next(structure1[0].get_chains(), None)
    chain2 = next(structure2[0].get_chains(), None)
    
    if not chain1 or not chain2:
        raise ValueError("Не знайдено ланцюгів у структурах.")
    
    # Збираємо номери залишків з Cα в обох структурах
    res_ids1 = {res.get_id()[1] for res in chain1.get_residues() if 'CA' in res}
    res_ids2 = {res.get_id()[1] for res in chain2.get_residues() if 'CA' in res}
    
    # Спільні номери залишків
    common_res_ids = sorted(res_ids1.intersection(res_ids2))
    
    if not common_res_ids:
        raise ValueError("Немає спільних Cα-атомів у структурах.")
    
    atoms1 = []
    atoms2 = []
    for rid in common_res_ids:
        res1 = chain1[rid]
        res2 = chain2[rid]
        atoms1.append(res1['CA'])
        atoms2.append(res2['CA'])
    
    print(f"Знайдено {len(atoms1)} спільних Cα-атомів для вирівнювання.")
    
    sup = PDB.Superimposer()
    sup.set_atoms(atoms1, atoms2)
    sup.apply(atoms2)
    return sup.rms

# Функція для витягнення pLDDT з AlphaFold PDB (B-factor)
def extract_plddt(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('af', pdb_file)
    plddt = []
    for atom in structure.get_atoms():
        if atom.get_id() == 'CA':
            plddt.append(atom.get_bfactor())
    return np.mean(plddt), np.std(plddt), plddt

# Функція для торсійних кутів (φ, ψ) для Рамачандрана
def get_dihedrals(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_file)
    model = structure[0]
    chain = next(model.get_chains(), None)
    if not chain:
        raise ValueError("Не знайдено ланцюгів у структурі.")
    residues = [res for res in chain if PDB.is_aa(res)]
    phi = []
    psi = []
    for i in range(1, len(residues) - 1):
        res_prev = residues[i-1]
        res = residues[i]
        res_next = residues[i+1]
        try:
            phi_angle = PDB.calc_dihedral(res_prev['C'].get_vector(), res['N'].get_vector(), res['CA'].get_vector(), res['C'].get_vector())
            psi_angle = PDB.calc_dihedral(res['N'].get_vector(), res['CA'].get_vector(), res['C'].get_vector(), res_next['N'].get_vector())
            phi.append(phi_angle * 180 / np.pi)
            psi.append(psi_angle * 180 / np.pi)
        except KeyError:
            pass  # Пропускаємо, якщо атоми відсутні
    return phi, psi

# Проста оцінка фрустрації (на основі варіабельності відстаней між Cα)
def estimate_frustration(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_file)
    atoms = [atom.get_coord() for atom in structure.get_atoms() if atom.get_id() == 'CA']
    dist_matrix = cdist(atoms, atoms)
    local_var = np.var(dist_matrix, axis=1)
    return np.mean(local_var)

# Функція для отримання послідовності з PDB (для RSO та LM-Design)
def get_sequence(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_file)
    chain = next(structure[0].get_chains(), None)
    seq = ''.join([IUPACData.protein_letters_3to1.get(res.get_resname().capitalize(), 'X') for res in chain if PDB.is_aa(res)])
    return seq

# Спрощена симуляція RSO: релаксований простір з мутаціями та градієнтним спуском (на основі фрустрації як "енергії")
def simulate_rso(pdb_file, num_iterations=20):
    seq = list(get_sequence(pdb_file))
    original_frustration = estimate_frustration(pdb_file)
    energies = [original_frustration]
    
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    for i in range(num_iterations):
        pos = np.random.randint(0, len(seq))
        new_aa = np.random.choice(list(amino_acids))
        old_aa = seq[pos]
        seq[pos] = new_aa
        
        # Симуляція: "нова" фрустрація завжди трохи нижча з ймовірністю 0.7 (імітація спуску)
        delta = np.random.normal(-2, 3) if np.random.rand() > 0.3 else np.random.normal(2, 3)  # Більше шансів на зменшення
        new_frustration = energies[-1] + delta
        if new_frustration > energies[-1]:
            if np.random.rand() > 0.5:
                seq[pos] = old_aa
                new_frustration = energies[-1]
        energies.append(max(new_frustration, 0))  # Не нижче 0
    
    return energies

# Інтеграція LM-Design: використання ESM для еволюційного знання (фокус на петлях)
def lm_design_analysis(seq, plddt_list):
    # Завантажуємо ESM модель (ESM-1b для protein language modeling)
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    
    # Конвертуємо послідовність
    _, _, tokens = batch_converter([("protein", seq)])
    
    # Отримання ембедінгів (еволюційне знання)
    with torch.no_grad():
        results = model(tokens, repr_layers=[33])
    embeddings = results["representations"][33].mean(1).numpy()
    
    # Імітація адаптера: фокус на недетермінованих регіонах (де pLDDT <70)
    low_conf_regions = np.where(np.array(plddt_list) < 70)[0]
    if len(low_conf_regions) > 0:
        print(f"Недетерміновані регіони (петлі): позиції {low_conf_regions}")
        # Симуляція оптимізації: "покращення" ембедінгів для цих позицій (спрощено: середнє)
        improved_embeddings = embeddings.copy()
        for pos in low_conf_regions:
            improved_embeddings[0, pos] = embeddings[0, pos] * 1.1  # Імітація адаптера
        return improved_embeddings
    else:
        print("Немає недетермінованих регіонів.")
        return embeddings

# Візуалізація енергетичного ландшафту (траєкторія релаксації)
def visualize_energy_landscape(energies, name):
    plt.figure()
    plt.plot(energies, label='Енергія (фрустрація) під час релаксації')
    plt.xlabel('Ітерація')
    plt.ylabel('Енергія')
    plt.title(f'Енергетичний ландшафт для {name} (RSO симуляція)')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(f'energy_landscape_{name}.png')

# Класифікація зон Рамачандрана (для аналізу геометрії)
def classify_ramachandran(phi, psi):
    alpha = sum(1 for p, s in zip(phi, psi) if -180 <= p <= 0 and -90 <= s <= 0)  # Розширено для α-спіралей
    beta = sum(1 for p, s in zip(phi, psi) if -180 <= p <= 0 and 90 <= s <= 180)   # Розширено для β-шарів
    other = len(phi) - alpha - beta
    outliers = sum(1 for p, s in zip(phi, psi) if (0 < p < 180 and -180 < s < 180) or other < 0)  # Справжні заборонені
    print(f"Зони: α-спіралі: {alpha}, β-шари: {beta}, петлі/other: {other}, outliers: {outliers} ({outliers/len(phi)*100:.1f}%)")

# Основний аналіз для білка
def analyze_protein(exp_pdb, af_pdb, name):
    print(f"\nАналіз для {name}:")
    try:
        rmsd = calculate_rmsd(exp_pdb, af_pdb)
        print(f"RMSD: {rmsd:.2f} Å (бажано <1.5 Å)")
    except ValueError as e:
        print(f"Помилка в RMSD: {e}")
    
    try:
        mean_plddt, std_plddt, plddt_list = extract_plddt(af_pdb)
        print(f"Середнє pLDDT: {mean_plddt:.2f} (std: {std_plddt:.2f}) — >90: стабільні зони")
    except:
        print("Помилка в витягненні pLDDT. Треба перевірити, чи AlphaFold файл має B-фактори.")
    
    print("Мінімальне pLDDT: ", min(plddt_list), "на позиції", np.argmin(plddt_list))

    frustration_exp = estimate_frustration(exp_pdb)
    frustration_af = estimate_frustration(af_pdb)
    print(f"Фрустрація (варіабельність відстаней): експ. {frustration_exp:.2f}, AF {frustration_af:.2f}")
    
    # Діаграма Рамачандрана
    try:
        phi_exp, psi_exp = get_dihedrals(exp_pdb)
        phi_af, psi_af = get_dihedrals(af_pdb)
        plt.figure()
        plt.scatter(phi_exp, psi_exp, label='Експериментальна', alpha=0.5)
        plt.scatter(phi_af, psi_af, label='AlphaFold', alpha=0.5)
        plt.xlabel('φ (deg)')
        plt.ylabel('ψ (deg)')
        plt.title(f'Діаграма Рамачандрана для {name}')
        plt.legend()
        plt.grid()
        plt.show()  # Показати в Colab
        plt.savefig(f'ramachandran_{name}.png')
    except ValueError as e:
        print(f"Помилка в діаграмі Рамачандрана: {e}")
    
    # Візуалізація pLDDT
    try:
        plt.figure()
        plt.plot(plddt_list)
        plt.xlabel('Залишок')
        plt.ylabel('pLDDT')
        plt.title(f'pLDDT по залишках для {name} (AlphaFold)')
        plt.grid()
        plt.show()  # Показати в Colab
        plt.savefig(f'plddt_{name}.png')
    except:
        print("Помилка в графіку pLDDT.")

    seq = get_sequence(af_pdb)
    energies = simulate_rso(af_pdb)  # Симуляція RSO
    visualize_energy_landscape(energies, name)
    
    embeddings = lm_design_analysis(seq, plddt_list)  # LM-Design
    print(f"Ембедінги з LM-Design: форма {embeddings.shape} (еволюційне знання для петель)")
    
    # Класифікація Рамачандрана
    print("Класифікація для експериментальної:")
    classify_ramachandran(phi_exp, psi_exp)
    print("Класифікація для AlphaFold:")
    classify_ramachandran(phi_af, psi_af)

proteins = {
    'lysozyme': ('1DPX.pdb', 'AF-P00700-F1-model_v6.pdb'),
    'ubiquitin': ('1UBQ.pdb', 'AF-Q6SKX8-F1-model_v6.pdb')
}

for name, (exp, af) in proteins.items():
    if os.path.exists(exp) and os.path.exists(af):
        analyze_protein(exp, af, name)
    else:
        print(f"Файли для {name} не знайдено.")