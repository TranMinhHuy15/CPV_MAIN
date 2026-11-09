#!/usr/bin/env bash
# RQ2 - Giai doan 0+1: cai goi he thong, tai du 2844 video Nexar, preflight (KHONG train). Log: /workspace/rq2_prep.log
W=/workspace
PKG=$W/RQ2_LSTM_RTX4080_Terminal_Checked
ZIP=$W/RQ2_LSTM_RTX4080_Terminal_Checked.zip
DATA=$W/Nexar_HF
ZIP_SHA=a7d7e3505667ec6e689a90bcb7e2ece06c1378451826f9f2e995afb83b8525e7
log(){ echo "[$(date '+%F %T')] $*"; }
log "START prep"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | grep -q 4080 || { log "FAIL: GPU khong phai RTX 4080/4080S - goi se tu choi"; exit 1; }
apt-get update -qq && apt-get install -y -qq ffmpeg libgl1 libglib2.0-0 tmux >/dev/null || { log "FAIL: apt-get"; exit 1; }
echo "$ZIP_SHA  $ZIP" | sha256sum -c - || { log "FAIL: ZIP sai hash hoac chua upload"; exit 1; }
[ -d "$PKG" ] || python3 -m zipfile -e "$ZIP" $W/
pip install -q huggingface_hub
for i in 1 2 3; do
  n=$(find "$DATA" -name "*.mp4" 2>/dev/null | wc -l)
  [ "$n" -eq 2844 ] && break
  log "Co $n/2844 video, tai (lan $i)"
  if [ "$i" -eq 1 ]; then X=0; else X=1; fi
  HF_HUB_DISABLE_XET=$X python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nexar-ai/nexar_collision_prediction', repo_type='dataset', local_dir='$DATA', allow_patterns=['train/*','test-public/*','test-private/*'], max_workers=8)"
done
n=$(find "$DATA" -name "*.mp4" | wc -l); log "So video: $n"; du -sh "$DATA"
[ "$n" -eq 2844 ] || { log "FAIL: thieu video"; exit 1; }
df -h $W | tail -1
log "PREFLIGHT seed42 (short + long), khong train"
python3 $PKG/run_rq2.py --data-root $DATA --seeds 42 --total-hours 4
rc=$?
log "preflight ma thoat $rc"
python3 $W/rq2_status.py
log "PREP_XONG - doc SMOKE o tren roi moi chay pilot"
