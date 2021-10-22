if [ $# != 2 ]; then
    echo "arguments error, arguments should be run.sh device_id  joint/bone/j_motion/b_motion"
    exit 1
fi

echo "Using device $1"
export CUDA_VISIBLE_DEVICES=$1
if [ "$2" == 'bone' ]; then
    echo "Using bone information"
    python -m paddle.distributed.launch main.py --work-dir ./work_dir/fsd_bone/ --bone True --motion False
elif [ "$2" == 'joint' ]; then
    echo "Using joint information"
    python -m paddle.distributed.launch main.py --work-dir ./work_dir/fsd_joint/ --bone False --motion False
elif [ "$2" == 'j_motion' ]; then
   echo "Using joint motion information"
   python -m paddle.distributed.launch main.py --work-dir ./work_dir/fsd_jmotion/ --bone False --motion True
elif [ "$2" == 'b_motion' ]; then
  echo "Using bone motion information"
  python -m paddle.distributed.launch main.py --work-dir ./work_dir/fsd_bmotion/ --bone True --motion True
fi
