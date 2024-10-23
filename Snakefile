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

rule time:
    input: "data/{dataset}/alignment.fasta",
           "data/{dataset}/tree.nwk"
    output: "data/{dataset}/time.txt"
    threads: 16
    shell: "time python optimize.py {input} rust >> {output} && time python optimize.py {input} pytorch >> {output}"