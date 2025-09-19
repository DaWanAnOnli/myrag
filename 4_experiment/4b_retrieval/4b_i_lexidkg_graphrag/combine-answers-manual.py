from pathlib import Path

dir_path = Path("../../../dataset/4_experiment/4b_experiment_answers/4b_i_lexidkg_graphrag").resolve()
out = dir_path / "graph_rag_answers_.jsonl"

# Pick only files starting with the prefix; exclude the output file itself
files = sorted(p for p in dir_path.glob("graph_rag_answers_*") if p.name != "graph_rag_answers_.jsonl")

line_count = 0
with out.open("w", encoding="utf-8") as fout:
    for p in files:
        with p.open("r", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():  # skip blank lines if any
                    fout.write(line)
                    line_count += 1

print(f"Merged {len(files)} files ({line_count} lines) into {out}")