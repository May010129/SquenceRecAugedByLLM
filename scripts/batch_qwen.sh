# 任务名称
JOB_NAME="LLMAug"

# 循环直到任务成功完成
while true; do
    echo "Submitting job..."
    
    # 使用 srun 提交任务
    srun -p s2_bigdata -N1 --gres=gpu:1 --cpus-per-task=16 --ntasks-per-node=1 --quotatype=auto --job-name=$JOB_NAME \
    python batch_inference.py
    
    # 检查任务的退出状态
    if [ $? -eq 0 ]; then
        echo "Job completed successfully."
        break
    else
        echo "Job was preempted or failed. Resubmitting..."
    fi
done
