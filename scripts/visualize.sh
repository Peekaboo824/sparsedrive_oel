export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/visualization/visualize.py \
	projects/configs/evopsf_sparsedrive_small_stage2.py \
	--result-path work_dirs/evopsf_sparsedrive_small_stage2/boston_sg_split_v3/test_TTT_boston_sg_split_v3_3/results.pkl \
	--out-dir vis/test_TTT_boston_sg_split_v3_3
