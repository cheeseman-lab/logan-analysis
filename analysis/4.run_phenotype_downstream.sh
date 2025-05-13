#!/bin/bash

RUN_DOWNSTREAM=True

# Run the SBS/phenotype rules
snakemake --use-conda --cores all \
    --snakefile "../brieflow/workflow/Snakefile" \
    --configfile "config/config.yml" \
    --rerun-triggers mtime \
    --until all_phenotype_downstream \
    --config run_downstream=$RUN_DOWNSTREAM