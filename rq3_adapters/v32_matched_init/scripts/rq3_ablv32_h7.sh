#!/usr/bin/env bash
# RQ3 ABLATION V3.2 (matched init, 6 arm x 3 seed x 50 epoch = 18 run) - NEN H7, Vast RTX4080-class.
#   Arm: C7 | R7 | A7 | A26 | U3 | G3 (tat ca train moi, khoi tao chung cung seed). KHONG dung ket qua V2 cho ket luan chinh.
#   May kia chay nen V6. Moi GPU chay mot goi.
#
# CACH CHAY (1 lenh, tu mo tmux 'rq3', chay het A->Z, log day du trong /workspace/rq3abl32_auto_*.log):
#   1) Dia may thue >= 150 GB. Neu Nexar chua co tren may: JupyterLab tao /workspace/hf_token.txt (1 dong = HF Read token)
#   2) FOLDER_URL='<link thu muc Drive co goi V3.2>' bash rq3_ablv32_h7.sh auto
#   Xem log:  cat /workspace/rq3abl32_auto_*.log     (theo doi: less +F <file>, Ctrl+C roi q de thoat)
#   Xem tmux (chi doc):  [ -z "$TMUX" ] && tmux attach -r -t rq3     (thoat: Ctrl+B roi D)
#   Giua chung se thay dong 'PILOT DONE' (sau khi extract + teacher 3 seed); neu pilot loi thi dung o do, cache van con.
# Lenh le: nexar | prep | go | run | pack
# Bien tuy chon: COMPACT=true (mac dinh; xoa *.pth/snapshot sau khi da xuat selected_adapter.npz + kiem hash, de tar nho tai duoc)
#                COMPACT=false (giu full checkpoint tren may; can o dia lon hon nhieu, tar van khong chua *.pth)
set -euo pipefail
MODE="${1:-prep}"
W=${W:-/workspace}
TAG=rq3abl32
DL=$W/${TAG}_dl
IN=$W/${TAG}_inputs
OUT=$W/outputs/rq3_ablv3_2_h7_42_43_44
RT=$W/runtime/rq3_checked
BUNDLE_NAME=RQ3_ABLV3_2_RECHECK_FIXED_20261107
BUNDLE_DIR=$W/$BUNDLE_NAME
PKG=$BUNDLE_DIR/RQ3_ABLV3_H7
DATA=${DATA:-$W/Nexar_HF}
FOLDER_URL=${FOLDER_URL:-}
COMPACT=${COMPACT:-true}
case "$COMPACT" in true|false) ;; *) echo "LOI: COMPACT phai la true hoac false" >&2; exit 1;; esac
SELF=$(readlink -f "$0")
TOKF=$W/hf_token.txt
if [ -z "${HF_TOKEN:-}" ] && [ -s "$TOKF" ]; then HF_TOKEN=$(tr -d ' \r\n\t' < "$TOKF"); export HF_TOKEN; fi
BUNDLE_SHA=631f7f3588da4e655353325bc5ebb463c2794cfd2a09b43c561b9dea8a2955fe
SRC_SHA=464a7635fe5e5c3d837078daab8cd42f4d657896f5602f80822c74f19a2a4220
PAYLOAD_SHA=3a5eb0c519129b4dd2b3040eec2726d1bf800bb92fabe39ff8aea3afdca23404
say(){ echo -e "\n=== $(date -u +%H:%M:%S)Z $* ==="; }
die(){ echo "LOI: $*" >&2; exit 1; }
pipi(){ python3 -m pip install -q "$@" 2>/dev/null || python3 -m pip install -q --break-system-packages "$@"; }
mp4n(){ find "$DATA" -name '*.mp4' 2>/dev/null | wc -l; }
one(){ local n; n=$(find "$DL" -type f -name "$1" | wc -l); [ "$n" -eq 1 ] || die "can dung 1 file '$1' trong $DL, thay $n"; find "$DL" -type f -name "$1"; }
launch(){ local LOG=$W/${TAG}_${1}_$(date +%m%d_%H%M).log
  command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
  TMUX= tmux kill-session -t rq3 2>/dev/null || true
  local E=(); [ -n "${3:-}" ] && E=(-e "$3")
  TMUX= tmux new-session -d -s rq3 "${E[@]}" "($2) 2>&1 | tee $LOG; RC=\${PIPESTATUS[0]}; echo; echo \"=== KET THUC: EXIT=\$RC (0 la xong, khac 0 la loi: xem $LOG) ===\" | tee -a $LOG; exec bash"
  echo "Dang chay trong tmux 'rq3'. Log day du: $LOG"
  echo "Xem:  cat $LOG    |  theo doi lien tuc: less +F $LOG  (Ctrl+C roi q de thoat, job van chay)"
  echo "Xem tmux (chi doc):  [ -z \"\$TMUX\" ] && tmux attach -r -t rq3   (thoat: Ctrl+B roi D)"; }

if [ "$MODE" = auto ]; then
  [ -n "$FOLDER_URL" ] || die "Thieu link Drive. Chay: FOLDER_URL='https://drive.google.com/drive/folders/...' bash $SELF auto"
  if [ "$(mp4n)" -ne 2844 ] && [ -z "${HF_TOKEN:-}" ]; then
    die "Nexar chua co va chua co HF token. JupyterLab: File > New > Text File, dan token (1 dong), Save As $TOKF, roi chay lai."
  fi
  ENVS=""; [ -n "${HF_TOKEN:-}" ] && ENVS="HF_TOKEN=$HF_TOKEN"   # token chi qua bien moi truong, khong vao log
  launch auto "bash $SELF nexar && FOLDER_URL='$FOLDER_URL' COMPACT=$COMPACT bash $SELF prep && COMPACT=$COMPACT bash $SELF go && bash $SELF pack" "$ENVS"
  exit 0
fi

if [ "$MODE" = nexar ]; then
  say "Nexar_HF"
  if [ "$(mp4n)" -eq 2844 ]; then echo "Da co du 2844 mp4 o $DATA -> bo qua"; exit 0; fi
  [ -n "${HF_TOKEN:-}" ] || die "Can HF token ($TOKF)"
  FREE=$(df -BG --output=avail $W | tail -1 | tr -dc 0-9); echo "Dia trong: ${FREE}G"; [ "$FREE" -ge 40 ] || die "Can >=40G trong de tai Nexar"
  pipi huggingface_hub
  mkdir -p "$DATA"
  HF_HUB_DISABLE_TELEMETRY=1 python3 -I - "$DATA" <<'PY' || die "Tai HF loi (token sai / chua Accept dataset / mang). Chay lai auto se tai tiep."
import os,sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id='nexar-ai/nexar_collision_prediction',repo_type='dataset',
    revision='7535d0656dac31d7da2846913bb69c6331d2a70a',local_dir=sys.argv[1],token=os.environ['HF_TOKEN'],
    max_workers=8,allow_patterns=['*.mp4','**/*.mp4'])
PY
  N=$(mp4n); echo "mp4=$N"; [ "$N" -eq 2844 ] || die "HF xong nhung chi co $N mp4 (can 2844); chay lai auto"
  rm -f "$TOKF"; echo "Nexar OK. Da xoa $TOKF"; exit 0
fi

if [ "$MODE" = prep ]; then
  say "May"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  FREE=$(df -BG --output=avail $W | tail -1 | tr -dc 0-9); echo "Dia trong: ${FREE}G"
  [ "$FREE" -ge "${MINFREE:-120}" ] || die "Can >=${MINFREE:-120}G trong (khuyen nghi thue dia >=150G; dat MINFREE=... de doi nguong)"
  need=(); for t in tmux unzip ffmpeg ffprobe; do command -v $t >/dev/null || need+=($t); done
  if [ ${#need[@]} -gt 0 ]; then apt-get update -qq && apt-get install -y -qq tmux unzip ffmpeg libgl1 libglib2.0-0; fi
  pipi gdown numpy scikit-learn

  say "Tai file tu Drive (goi V3.2, archive.zip, H7_BACKUP_ALL.zip)"
  mkdir -p "$DL"
  if [ -z "$(find "$DL" -type f -name "$BUNDLE_NAME.zip")" ] || [ -z "$(find "$DL" -type f -name 'H7_BACKUP_ALL.zip')" ]; then
    [ -n "$FOLDER_URL" ] || die "Thieu FOLDER_URL"
    gdown --folder "$FOLDER_URL" -O "$DL" || die "gdown loi (thu muc chua share dung cach hoac Drive gioi han tai)"
  fi
  BZ=$(one "$BUNDLE_NAME.zip")
  AZ=$(one 'archive.zip'); H7Z=$(one 'H7_BACKUP_ALL.zip')

  say "Kiem hash"
  [ "$(sha256sum "$BZ" | cut -d' ' -f1)" = "$BUNDLE_SHA" ] || die "Goi $BUNDLE_NAME.zip sai hash (ban cu / tai hong?)"
  [ -d "$BUNDLE_DIR" ] || unzip -q "$BZ" -d "$W"
  [ -d "$PKG" ] || die "thieu $PKG sau khi giai nen"
  SRC=$BUNDLE_DIR/inputs/rq3_selected_source_v1.zip
  [ "$(sha256sum "$SRC" | cut -d' ' -f1)" = "$SRC_SHA" ] || die "rq3_selected_source_v1.zip sai hash"
  echo "goi V3.2 OK, source OK"
  mkdir -p "$IN"
  [ -d "$IN/payload" ] || unzip -q "$AZ" -d "$IN/payload"
  PAY=$(dirname "$(find "$IN/payload" -name RQ3_H7_SELECTED_PAYLOAD.json)")
  python3 -I - "$PAY" "$PAYLOAD_SHA" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,expected=Path(sys.argv[1]),sys.argv[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(root/'RQ3_H7_SELECTED_PAYLOAD.json')==expected,'payload manifest sai hash'
m=json.loads((root/'RQ3_H7_SELECTED_PAYLOAD.json').read_text())
bad=[n for n,h in m['files'].items() if sha(root/n)!=h]
assert not bad,'payload file sai hash: %s'%bad
print('payload OK:',len(m['files']),'file')
PY

  say "Dung H7 seed 42/43/44 (stage_h7.py cua payload, giong V2/v1.1)"
  H7SRC=$W/${TAG}_h7src
  mkdir -p "$H7SRC"; [ -e "$H7SRC/H7_BACKUP_ALL.zip" ] || ln "$H7Z" "$H7SRC/H7_BACKUP_ALL.zip" 2>/dev/null || cp "$H7Z" "$H7SRC/"
  if [ -z "$(find "$IN/H7" -maxdepth 1 -type d -name medium_h7 2>/dev/null)" ]; then
  python3 -I - "$PAY" "$H7SRC" "$IN/H7" <<'PY'
import importlib.util,sys,json
from pathlib import Path
pay,ck,out=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('stage_h7',pay/'stage_h7.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
runs=m.materialize_h7_exports(ck,pay/'baseline_metadata',out)
print(json.dumps(runs,indent=1))
PY
  fi
  H7S=$(find "$IN/H7" -maxdepth 1 -type d -name medium_h7); [ -d "$H7S" ] || die "khong thay $IN/H7/medium_h7"
  for s in 42 43 44; do [ -d "$H7S/seed_$s/run" ] || die "thieu $H7S/seed_$s/run"; done

  say "Dataset"
  for d in train test-public test-private; do [ -d "$DATA/$d" ] || die "thieu $DATA/$d"; done
  N=$(mp4n); echo "mp4=$N"; [ "$N" -eq 2844 ] || die "Nexar chua du 2844 mp4"

  say "inputs.json (prior_study_root=null, cache_roots=null: extract moi, compact=$COMPACT)"
  python3 -I - "$PKG" "$IN" "$H7S" "$DATA" "$OUT" "$RT" "$SRC" "$COMPACT" <<'PY'
import json,sys
pkg,inp,h7,data,out,rt,src,compact=sys.argv[1:]
c=json.load(open(pkg+'/inputs.example.json'))
c.update(source_zip=src,data_root=data,output_root=out,runtime_root=rt,
         h7_runs={s:'%s/seed_%s/run'%(h7,s) for s in ('42','43','44')},
         prior_study_root=None,cache_roots={'42':None,'43':None,'44':None},compact_completed_trials=(compact=='true'))
json.dump(c,open(pkg+'/inputs.json','w'),indent=2);print(json.dumps(c,indent=2))
PY
  say "preflight (kiem 3 base + source + dataset, chua train)"
  cd "$PKG" && python3 launch.py --config inputs.json preflight
  echo -e "\nPREP XONG (preflight qua)."

elif [ "$MODE" = go ]; then
  [ -f "$PKG/inputs.json" ] || die "chua prep"
  say "Hash ke hoach phan tich (ghi truoc khi train) -> $W/${TAG}_preregistration_sha256.txt"
  sha256sum "$PKG/PREREGISTERED_ANALYSIS.md" "$PKG/bootstrap_paired.py" "$PKG/ablation_stats.py" "$PKG/protocol.py" "$PKG/analyze_cross_base.py" | tee $W/${TAG}_preregistration_sha256.txt
  cd "$PKG"
  say "launch.py all: setup -> extract(3 seed) + teacher -> PILOT (C7 seed42, epoch 2->3) -> train 18 run -> freeze -> test -> bootstrap"
  python3 launch.py --config inputs.json all
  say "HOAN TAT TINH TOAN"
  ls -l "$OUT/comparison" 2>/dev/null || true

elif [ "$MODE" = run ]; then
  launch run "COMPACT=$COMPACT bash $SELF go && bash $SELF pack"

elif [ "$MODE" = pack ]; then
  [ -f "$OUT/comparison/BOOTSTRAP_COMPLETE.json" ] || echo "CANH BAO: chua co comparison/BOOTSTRAP_COMPLETE.json (dong goi phan dang co)"
  cd $W; N=rq3_ablv32_h7_RESULTS_$(date +%Y%m%d_%H%M); E=evidence_$N; R=RQ3ABLV32_H7_RESULT; mkdir -p "$E" $R
  O=outputs/rq3_ablv3_2_h7_42_43_44
  say "Bang chung chay -> $E"
  nvidia-smi > $E/nvidia_smi.txt; nvidia-smi -q > $E/nvidia_smi_full.txt
  python3 -m pip freeze > $E/pip_freeze_host.txt 2>/dev/null || true
  [ -x "$RT/venv/bin/python" ] && "$RT/venv/bin/python" -m pip freeze > $E/pip_freeze_runtime.txt 2>/dev/null || true
  { date -u; hostname; uname -a; df -h $W; free -h; nproc; echo "COMPACT=$COMPACT"; } > $E/system_info.txt
  cp "$SELF" $E/
  find $O -path '*/caches' -prune -o -type f -printf '%10s  %p\n' | sort -k2 > $E/FILE_LIST.txt
  find $O -path '*/caches' -prune -o -type f -print0 | sort -z | xargs -0 sha256sum > $E/SHA256SUMS_outputs.txt
  say "Tar day du (khong cache feature, khong *.pth / snapshot)"
  tar -cf $R/$N.tar --exclude='caches' --exclude='*.npz.partial' --exclude='__pycache__' --exclude='tests/scratch' \
      --exclude='*.pth' --exclude='safe_resume_snapshot.zip' \
      $O $BUNDLE_NAME "$E" $(ls ${TAG}_*.log ${TAG}_preregistration_sha256.txt 2>/dev/null)
  (cd $R && sha256sum $N.tar | tee $N.tar.sha256)
  say "Zip log + metrics (moi file .log/.csv/.json/.jsonl/.md/.txt)"
  python3 -I - "$O" "$E" "$R" "$TAG" RQ3ABLV32_H7 <<'PY'
import os,sys,zipfile
o,e,r,tag,name=sys.argv[1:];ext=('.log','.csv','.json','.jsonl','.md','.txt');c=0
srcs=[o,e]+sorted(f for f in os.listdir('.') if f.startswith(tag+'_') and f.endswith(('.log','.txt')))
with zipfile.ZipFile('%s/%s_LOGS_METRICS_ALL.zip'%(r,name),'w',zipfile.ZIP_DEFLATED) as z:
    for s in srcs:
        if os.path.isfile(s):z.write(s);c+=1;continue
        for root,dirs,files in os.walk(s):
            dirs[:]=[d for d in dirs if d!='caches']
            for f in files:
                if f.endswith(ext):z.write(os.path.join(root,f));c+=1
with zipfile.ZipFile('%s/%s_RESULTS_SUMMARY.zip'%(r,name),'w',zipfile.ZIP_DEFLATED) as z:
    comp=o+'/comparison'
    if os.path.isdir(comp):
        for f in sorted(os.listdir(comp)):
            p=os.path.join(comp,f)
            if os.path.isfile(p):z.write(p,f)
    for s in ('42','43','44'):
        for d in ('teachers','teacher'):
            p='%s/%s/seed_%s'%(o,d,s)
            if os.path.isdir(p):
                for f in sorted(os.listdir(p)):
                    if f.endswith('.json'):z.write(os.path.join(p,f),'%s_seed%s_%s'%(d,s,f))
    for f in ('%s_preregistration_sha256.txt'%tag,):
        if os.path.isfile(f):z.write(f)
print('zip log+metrics:',c,'file')
PY
  (cd $R && sha256sum *.zip | tee -a $N.tar.sha256 >/dev/null; ls -lh; echo; cat $N.tar.sha256)
  echo -e "\nTAI VE 4 FILE trong $W/$R/ (JupyterLab: tick tung file -> Download), roi certutil so hash."
else die "mode: auto | nexar | prep | go | run | pack"; fi
