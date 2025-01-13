srun -p s2_bigdata --gres=gpu:0 -n1 -N1 --quotatype==spot \
    python batch_inference.py