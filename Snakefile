rule download:
    output: "data/{dataset}/alignment.sto"
    shell: "wget -O - https://www.ebi.ac.uk/interpro/api/entry/pfam/{wildcards.dataset}?annotation=alignment:seed | gunzip -c > {output}"

rule fasta:
    input: "data/{dataset}/alignment.sto"
    output: "data/{dataset}/alignment.fasta"
    shell: "esl-reformat afa {input} > {output}"

rule phylo:
    input: "data/{dataset}/alignment.fasta"
    output: "data/{dataset}/tree.nwk"
    shell: "./FastTreeDbl {input} > {output}"

rule random_phylo:
    output: "data/random/root_tree_{num_leafs}.nwk"
    shell: "phylotree generate -t {wildcards.num_leafs} -b > {output}"

rule unroot:
    input: "data/random/root_tree_{num_leafs}.nwk"
    output: "data/random/tree_{num_leafs}.nwk"
    shell: "nw_reroot -d {input} > {output}"

rule random_fasta:
    input: "data/random/tree_{num_leafs}.nwk"
    output: "data/random/alignment_{num_leafs}_{L}.fasta"
    shell: "python random_fasta.py {input[0]} {wildcards.L} {output[0]}"

def benchmark_input(wildcards):
    num_leafs = [10,20,30,100,500,1000,2000]
    L = [300] * len(num_leafs)

    newick = [f"data/random/tree_{n}.nwk" for n in num_leafs]
    fasta = [f"data/random/alignment_{n}_{l}.fasta" for n,l in zip(num_leafs,L)]

    return newick + fasta

rule benchmark:
    input: files = benchmark_input, script = "benchmark.py"
    output: "data/random/time_{num_t}.txt"
    threads: {num_t}
    resources:
        slurm_extra="--exclusive",
        mem_mb=800000,
        runtime=1000
    shell: "OMP_NUM_THREADS={threads} RAYON_NUM_THREADS={threads} python benchmark.py {input.files} > {output}"

rule time:
    input: "data/{dataset}/alignment.fasta",
           "data/{dataset}/tree.nwk"
    output: "data/{dataset}/time.txt"
    threads: 16
    resources:
        mem_mb = 100000,
        runtime = 600
    shell: "/usr/bin/time -v python optimize.py {input} rust >> {output} && /usr/bin/time -v python optimize.py {input} pytorch >> {output}"
