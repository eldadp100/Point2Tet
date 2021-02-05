# name from filename
NAME=$0
NAME=${NAME##*/}
NAME=${NAME%.*}
NAME=${NAME#eval_}

python full_eval.py --indir datasets --outdir results --modeldir models --dataset base/testset.txt --models p2s_max --modelpostfix _model_249.pth --batchSize 501 --workers 7 --cache_capacity 5 --query_grid_resolution 256 --epsilon 3 --certainty_threshold 13 --sigma 5
