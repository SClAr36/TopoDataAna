from utils.load_cif import load_cif, summarize_elements, measure_molecule_size


weight_dict = {'C':6, 'N':7, 'O':8, 'S':16, 'P':15, 'F':9, 'CL':17, 'BR':35, 'I':53,}

coords, weights = load_cif("data/biomole/4QG3.cif", "rna", mode="rna_only", output="weights")




