import os
import argparse

# 23 species IDs, same order as iPro-MP README
SPECIES_NAMES = [
    "Acinetobacter_baumannii_ATCC_17978",
    "Bradyrhizobium_japonicum_USDA_110",
    "Burkholderia_cenocepacia_J2315",
    "Campylobacter_jejuni_RM1221",
    "Campylobacter_jejuni_81116",
    "Campylobacter_jejuni_81-176",
    "Campylobacter_jejuni_NCTC_11168",
    "Corynebacterium_diphtheriae_NCTC_13129",
    "Corynebacterium_glutamicum_ATCC_13032",
    "Escherichia_coli_K-12_MG1655",
    "Haloferax_volcanii_DS2",
    "Helicobacter_pylori_26695",
    "Nostoc_sp_PCC7120",
    "Paenibacillus_riograndensis_SBR5",
    "Pseudomonas_putida_KT2440",
    "Shigella_flexneri_M90T",
    "Sinorhizobium_meliloti_1021",
    "Staphylococcus_aureus_MW2",
    "Staphylococcus_epidermidis_ATCC_12228",
    "Synechococcus_elongatus_PCC_7942",
    "Thermococcus_kodakarensis_KOD1",
    "Xanthomonas_campestris_B100",
    "Bacillus_subtilis_168",
]


def parse_fasta(path):
    header = None
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('>'):
                header = line
            else:
                is_neg = (' CDS' in header) or (' Convergent' in header)
                yield line, (0 if is_neg else 1)
                header = None


def write_data_file(out_path, records):
    with open(out_path, 'w') as f:
        for i, (seq, lab) in enumerate(records):
            f.write(f"{i}\t{seq}\t{lab}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ipromp_root", type=str,
        default=os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "Benchmark Dataset")),
        help="iPro-MP 'Benchmark Dataset' folder (contains Train/ and Test/)")
    ap.add_argument(
        "--out_root", type=str,
        default=os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "Datasets_dfm")),
        help="Where to write our-style layout")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    fileslist = []

    for sid in range(1, 24):
        ds_name = f"species_{sid:02d}_{SPECIES_NAMES[sid - 1]}"
        dst_dir = os.path.join(args.out_root, ds_name)
        os.makedirs(dst_dir, exist_ok=True)

        for split_in, split_out in [("Train", "train"), ("Test", "test")]:
            src = os.path.join(args.ipromp_root, split_in, f"{sid}_{split_in.lower()}.txt")
            dst = os.path.join(dst_dir, f"{split_out}.data")
            records = list(parse_fasta(src))
            write_data_file(dst, records)
            pos = sum(1 for _, l in records if l == 1)
            neg = sum(1 for _, l in records if l == 0)
            print(f"[{ds_name}] {split_out}.data  N={len(records)}  pos={pos}  neg={neg}")

        fileslist.append(ds_name)

    with open(os.path.join(args.out_root, "FILESLIST.txt"), 'w') as f:
        for name in fileslist:
            f.write(name + "\n")
    print(f"Wrote FILESLIST.txt with {len(fileslist)} species -> {args.out_root}")


if __name__ == "__main__":
    main()
