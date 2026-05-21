from collections import defaultdict

def build_vocab(sequences):
    technique_set = set()
    tactic_set = set()
    intrusion_set = set()

    for group in sequences:
        intrusion_set.add(group["intrusion_set"])
        for step in group["sequence"]:
            technique_set.add(step["technique"])
            tactic_set.add(step["tactic"])

    technique2idx = {t: i + 1 for i, t in enumerate(sorted(technique_set))}
    tactic2idx = {t: i + 1 for i, t in enumerate(sorted(tactic_set))}
    intrusion2idx = {g: i + 1 for i, g in enumerate(sorted(intrusion_set))}

    # 0 = PAD/UNK
    technique2idx["<UNK>"] = 0
    tactic2idx["<UNK>"] = 0
    intrusion2idx["<UNK>"] = 0

    return technique2idx, tactic2idx, intrusion2idx
