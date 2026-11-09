#!/usr/bin/env bash
set -euo pipefail
cd /workspace
ZIP=RQ2_Medium_H13_Checked_v1.zip
WANT=f0757a04cad06c95e2bc64f3949a02f48d044bb117751ab5f8e13e2397cdca22

# Neu chua o trong tmux thi tu mo tmux roi chay lai chinh script nay (giu foreground)
if [ -z "${TMUX:-}" ]; then
  command -v tmux >/dev/null || { apt-get update && apt-get install -y tmux; }
  exec tmux new-session -A -s rq2_h13 "bash /workspace/start_h13.sh; echo; echo 'Script ket thuc. Nhan Enter de dong.'; read"
fi

echo "== 1/4 kiem hash zip"
GOT=$(sha256sum "$ZIP" | cut -d' ' -f1)
if [ "$GOT" != "$WANT" ]; then echo "HASH KHONG KHOP: $GOT"; echo "Dung lai, upload lai zip."; exit 1; fi
echo "hash OK"

echo "== 2/4 cai cong cu he thong"
apt-get update -qq && apt-get install -y -qq tmux unzip ffmpeg libgl1 libglib2.0-0

echo "== 3/4 giai nen"
[ -d RQ2_Medium_H13_Checked_v1 ] || unzip -q "$ZIP"
python3 /workspace/RQ2_Medium_H13_Checked_v1/medium_ops.py verify

echo "== 4/4 chay H13 (tai data -> preflight -> smoke GPU -> train 3 seed)"
echo "Neu duoc hoi 'HF Read token', dan token MOI vao o an roi Enter."
HOURS=72 bash /workspace/RQ2_Medium_H13_Checked_v1/run_auto.sh
