#!/bin/bash
cd /path/to/project
say(){ echo "[$(date +%H:%M:%S)] $*"; }
python3 mk_thitogene.py || exit 1
grep -n "add_argument(\"--fold\"\|--fold" src/thitogene_hest.py | head -2
say "冒烟：SKCM fold0（GPU）"
cat > jobs/thito_smoke.sh <<'SH'
#!/bin/bash
#SBATCH -J thitosmoke
#SBATCH --qos=YOUR_QOS --partition=YOUR_GPU_PARTITION --gres=gpu:1 -c 6 --mem=64G -t 4:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1
python -u src/thitogene_hest.py --cohorts SKCM --fold 0 --out results/thito_smoke.json
SH
J=$(sbatch --parsable jobs/thito_smoke.sh); until [ $(squeue -j $J -h|wc -l) -eq 0 ]; do sleep 30; done
tail -8 logs/thitosmoke_${J}.out
[ -s results/thito_smoke.json ] || { say "冒烟失败"; exit 1; }
say "全量：29 折数组"
cp jobs/hggep_arr.sh jobs/thito_arr.sh 2>/dev/null || { ls jobs | grep -i hggep; exit 1; }
sed -i "s/hggep_hest.py/thitogene_hest.py/g; s/-J hggep[a-z_]*/-J thito/; s/hggep_hest/thitogene_hest/g; s/hggep_f/thito_f/g" jobs/thito_arr.sh
grep -n "python\|--out\|#SBATCH -J\|partition" jobs/thito_arr.sh | head -6
sbatch jobs/thito_arr.sh >/dev/null
until [ $(squeue -u $USER -r -h -n thito | wc -l) -eq 0 ]; do sleep 120; done
ls results/ | grep -c "thitogene_hest" ; say "29 折完成"
