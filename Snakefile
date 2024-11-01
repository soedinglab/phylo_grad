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
    output: "data/random/tree_{num_leafs}.nwk"
    shell: "ngesh -L {wildcards.num_leafs} --seed 42 > {output}"

rule random_fasta:
    input: "data/random/tree_{num_leafs}.nwk"
    output: "data/random/alignment_{num_leafs}_{L}.fasta"
    shell: "python random_fasta.py {input[0]} {wildcards.L} {output[0]}"

rule time:
    input: "data/{dataset}/alignment.fasta",
           "data/{dataset}/tree.nwk"
    output: "data/{dataset}/time.txt"
    threads: 16
    resources:
        mem_mb = 100000,
        runtime = 600
    shell: "/usr/bin/time -v python optimize.py {input} rust >> {output} && /usr/bin/time -v python optimize.py {input} pytorch >> {output}"
