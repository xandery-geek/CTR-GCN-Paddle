joint_dir="fsd_augm_joint"
bone_dir="fsd_augm_bone"
jmotion_dir="fsd_augm_jmotion"
bmotion_dir="fsd_augm_bmotion"

joint_pdparams=""
bone_pdparams=""
jmotion_pdparams=""
bmotion_pdparams=""

if [ $# == 1 ]; then
  command=$1
  device=0
elif [ $# == 2 ]; then
  command=$1
  device=$2
else
  echo "arguments error!"
  exit 1
fi

echo "Using device ${device}"
export CUDA_VISIBLE_DEVICES=$1

if [ "${command}" == 'joint' ]; then
  echo "Using joint information"
  python main.py --work-dir ./work_dir/${joint_dir}/ --bone False --motion False
elif [ "${command}" == 'bone' ]; then
  echo "Using bone information"
  python main.py --work-dir ./work_dir/${bone_dir}/ --bone True --motion False
elif [ "${command}" == 'j_motion' ]; then
   echo "Using joint motion information"
   python main.py --work-dir ./work_dir/${jmotion_dir}/ --bone False --motion True
elif [ "${command}" == 'b_motion' ]; then
  echo "Using bone motion information"
  python main.py --work-dir ./work_dir/${bmotion_dir}/ --bone True --motion True

elif [ "${command}" == 'all' ]; then
  echo "Using joint information"
  python main.py --work-dir ./work_dir/${joint_dir}/ --bone False --motion False

  echo "Using bone information"
  python main.py --work-dir ./work_dir/${bone_dir}/ --bone True --motion False

  echo "Using joint motion information"
  python main.py --work-dir ./work_dir/${jmotion_dir}/ --bone False --motion True

  echo "Using bone motion information"
  python main.py --work-dir ./work_dir/${bmotion_dir}/ --bone True --motion True

elif [ "${command}" == 'predict' ]; then
  echo "predict with joint information"
  python main.py --phase predict --work-dir ./work_dir/${joint_dir}/ --bone False --motion False --weights ./work_dir/${joint_dir}/"${joint_pdparams}"
  echo "predict with bone information"
  python main.py --phase predict --work-dir ./work_dir/${bone_dir}/ --bone True --motion False --weights ./work_dir/${bone_dir}/"${bone_pdparams}"
  echo "predict with joint and motion information"
  python main.py --phase predict --work-dir ./work_dir/${jmotion_dir}/ --bone False --motion True --weights ./work_dir/${jmotion_dir}/"${jmotion_pdparams}"
  echo "predict with bone and motion information"
  python main.py --phase predict --work-dir ./work_dir/${bmotion_dir}/ --bone True --motion True --weights ./work_dir/${bmotion_dir}/"${bmotion_pdparams}"

elif [ "${command}" == 'ensemble' ]; then
  python ensemble.py --phase predict --joint-dir work_dir/${joint_dir}/ --bone-dir work_dir/${bone_dir}/ --joint-motion-dir work_dir/${jmotion_dir}/ --bone-motion-dir work_dir/${bmotion_dir}/
fi
