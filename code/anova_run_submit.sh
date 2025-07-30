#!/bin/bash

for portion_index in {0..7}
do
    sbatch anova_run.sh "$portion_index"
done