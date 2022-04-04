dvc run \
-n fix_spaces \
-d data/raw/census.csv \
-o data/processed/census-fix-spaces.csv \
"sed 's/, /,/g' data/raw/census.csv > data/processed/census-fix-spaces.csv"

dvc run \
-n remove_dups \
-d data/processed/census-fix-spaces.csv \
-o data/processed/census-fix-spaces-nodups.csv \
"awk '{counts[$0]++;if (counts[$0] == 1) {print $0}}' data/processed/census-fix-spaces.csv > data/processed/census-fix-spaces-nodups.csv"
