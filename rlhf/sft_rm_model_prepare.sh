SOURCE_RM_PATH=$1
SOURCE_SFT_PATH=$2
TARGET_RM_PATH=$3
TARGET_SFT_PATH=$4
TENSOR_PARALLEL_SIZE=$5

TARGET_RM_TRAIN_PATH=${TARGET_RM_PATH}/rw_train
TARGET_RM_PRED_PATH=${TARGET_RM_PATH}/rw_pred
TARGET_SFT_TRAIN_PATH=${TARGET_SFT_PATH}/sft_train
TARGET_SFT_PRED_PATH=${TARGET_SFT_PATH}/sft_pred

mkdir -p ${TARGET_RM_TRAIN_PATH}
mkdir -p ${TARGET_RM_PRED_PATH}
mkdir -p ${TARGET_SFT_TRAIN_PATH}
mkdir -p ${TARGET_SFT_PRED_PATH}

cp -r ${SOURCE_RM_PATH}/* ${TARGET_RM_TRAIN_PATH}/
for (( i=0; i<TENSOR_PARALLEL_SIZE; i++ ))
do
    id=`echo ${i}|awk '{printf("%02d\n",$0)}'`
    cp -r ${TARGET_RM_TRAIN_PATH}/release/mp_rank_${id} ${TARGET_RM_TRAIN_PATH}/release/mp_rank_${id}_000
    cp ${TARGET_RM_TRAIN_PATH}/release/mp_rank_${id}_000/model_rng.pt ${TARGET_RM_TRAIN_PATH}/release/mp_rank_${id}_000/optim.pt
    echo "finish 1"
done
echo "rw_train finished"

cp -r ${SOURCE_SFT_PATH}/* ${TARGET_SFT_TRAIN_PATH}/
for (( i=0; i<TENSOR_PARALLEL_SIZE; i++ ))
do
    id=`echo ${i}|awk '{printf("%02d\n",$0)}'`
    cp -r ${TARGET_SFT_TRAIN_PATH}/release/mp_rank_${id} ${TARGET_SFT_TRAIN_PATH}/release/mp_rank_${id}_000
    cp ${TARGET_SFT_TRAIN_PATH}/release/mp_rank_${id}_000/model_rng.pt ${TARGET_SFT_TRAIN_PATH}/release/mp_rank_${id}_000/optim.pt
    echo "finish 1"
done
echo "sft_train finished"

cp -r ${SOURCE_RM_PATH}/* ${TARGET_RM_PRED_PATH}/
for (( i=0; i<TENSOR_PARALLEL_SIZE; i++ ))
do
    id=`echo ${i}|awk '{printf("%02d\n",$0)}'`
    mv ${TARGET_RM_PRED_PATH}/release/mp_rank_${id}/model_rng.pt ${TARGET_RM_PRED_PATH}/release/mp_rank_${id}/model_optim_rng.pt
    echo "finish 1"
done
echo "rw_pred finished"

cp -r ${SOURCE_SFT_PATH}/* ${TARGET_SFT_PRED_PATH}/
for (( i=0; i<TENSOR_PARALLEL_SIZE; i++ ))
do
    id=`echo ${i}|awk '{printf("%02d\n",$0)}'`
    mv ${TARGET_SFT_PRED_PATH}/release/mp_rank_${id}/model_rng.pt ${TARGET_SFT_PRED_PATH}/release/mp_rank_${id}/model_optim_rng.pt
    echo "finish 1"
done
echo "sft_pred finished"
