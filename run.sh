scenes=(
# 17DRP5sb8fy
# 1LXtFkjw3qL
# 1pXnuDYAj8r
# 29hnd4uzFmX
2azQ1b91cZZ
# 2t7WUuJeko7
# 5ZKStnWn8Zo
# ARNzJeq3xxb
# RPmz2sHmrrY
# Vt2qJdWjCF2
# WYY7iVyf5p8
# YFuZgdQ5vWj
# YVUC4YcDtcY
# fzynW3qQPVF
# gYvKGZ5eRqb
# gxdoqLR6rwA
# jtcxE69GiFV
# pa4otMbVnkk
# q9vSo1VnCiC
# rqfALeAoiTq
)

# scenes to test again 1LXtFkjw3qL, YFuZgdQ5vWj

for scene in ${scenes[@]}; do
    echo "Running evaluation for scene: $scene"
    python run_aeqa_evaluation.py --cf cfg/eval_episodiceqa.yaml --scene_id $scene

done