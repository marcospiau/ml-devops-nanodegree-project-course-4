# Fix spaces
dvc run \
-n fix_spaces \
-d data/raw/census.csv \
-o data/processed/census-fix-spaces.csv \
"sed 's/, /,/g' data/raw/census.csv > data/processed/census-fix-spaces.csv"

# Remove duplicates
# CUIDADO, esse 0 eh susbtituido por -zsh. Talvez usar aspas duplas resolva
dvc run \
-n remove_dups \
-d data/processed/census-fix-spaces.csv \
-o data/processed/census-fix-spaces-nodups.csv \
"awk '{counts[$0]++;if (counts[$0] == 1) {print $0}}' data/processed/census-fix-spaces.csv > data/processed/census-fix-spaces-nodups.csv"

# Prepare data and train model
dvc run \
-n train_eval_model \
-d data/processed/census-fix-spaces-nodups.csv \
-o data/train_test_data/X_train.joblib \
-o data/train_test_data/X_test.joblib \
-o model/encoder.joblib \
-o model/lb.joblib \
-o model/model.joblib \
-o model/slice_output.txt \
-M model/summary.json \
python3 starter/train_model.py
