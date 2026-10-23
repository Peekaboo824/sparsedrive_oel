export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

#python tools/data_converter/nuscenes_converter.py nuscenes \
#    --root-path ./data/nuscenes \
#    --canbus ./data/nuscenes \
#    --out-dir ./data/infos/ \
#    --extra-tag nuscenes \
#    --version v1.0-mini

#python tools/data_converter/nuscenes_converter.py nuscenes \
#    --root-path  /root/autodl-tmp/SparseDrive/data/nuscenes \
#    --canbus /root/autodl-tmp/SparseDrive/data/nuscenes \
#    --out-dir /root/autodl-tmp/SparseDrive/data/infos/ \
#    --extra-tag nuscenes \
#    --version v1.0

python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path  /root/autodl-tmp/SparseDrive/data/nuscenes \
    --canbus /root/autodl-tmp/SparseDrive/data/nuscenes \
    --out-dir /root/autodl-tmp/SparseDrive_OEL_2/data/infos/custom_split_sg_8_2\
    --extra-tag nuscenes_sg \
    --version v1.0

#python tools/data_converter/nuscenes_converter.py nuscenes \
#    --root-path /root/autodl-tmp/SparseDrive/data/nuscenes \
#    --canbus /root/autodl-tmp/SparseDrive/data/nuscenes \
#    --version v1.0 \
#    --out-dir /root/autodl-tmp/SparseDrive/data/infos/nuscenes_longtail_val_only \
#    --extra-tag nuscenes_longtail \
#    --filter-val-longtail \
#    --val-only