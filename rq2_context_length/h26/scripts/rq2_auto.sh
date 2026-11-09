#!/usr/bin/env bash
# RQ2 TU DONG, HIEN TRUC TIEP: cho prep -> kiem SMOKE -> pilot seed42 (chi val) -> quy tac quyet dinh
#   -> DAT: chay du 3 seed (FULL=1) | KHONG DAT: mo test seed42 roi dung -> backup LITE + BEST ckpt + sha256.
# Bi ngat (loi, het tien, Ctrl+C): chay lai DUNG lenh cu, phan da xong tu bo qua. Script KHONG tat/destroy may.
W=${W:-/workspace}
PKG=$W/RQ2_LSTM_RTX4080_Terminal_Checked
DATA=$W/Nexar_HF
OUT=$W/rq2_lstm_checked
FULL=${FULL:-1}
PILOT_H=${PILOT_H:-60}
FULL_H=${FULL_H:-150}
STATE=$W/rq2_auto_state.txt
log(){ echo; echo "[$(date '+%F %T')] ===== $* ====="; echo "$(date '+%F %T') $*" >> "$STATE"; }
die(){ log "DUNG: $*"; echo "Cuon len xem loi. Sua xong chay lai dung lenh cu - phan da xong se duoc bo qua."; exit 1; }
trap 'echo; echo "Ctrl+C: dung script. Chay lai dung lenh cu de tiep tuc."; exit 130' INT

[ -n "$TMUX" ] && tmux rename-window auto 2>/dev/null
echo "RQ2 AUTO | FULL=$FULL | cuon: lan chuot hoac Ctrl+B [ (q de thoat) | thoat ma van chay: Ctrl+B D | chon cua so: Ctrl+B W"

run_q(){ # $1 = ten giai doan, con lai = tham so cho run_rq2.py; tu chay lai toi da 3 lan (resume tu snapshot)
  local name=$1; shift
  local i rc
  for i in 1 2 3; do
    log "$name - lan $i: run_rq2.py $*"
    python3 "$PKG/run_rq2.py" --data-root "$DATA" "$@" 2>&1 | tee -a "$W/rq2_run.log"
    rc=${PIPESTATUS[0]}
    log "$name - ma thoat $rc"
    [ "$rc" -eq 0 ] && return 0
    [ "$i" -lt 3 ] && { echo "Loi/ngat -> cho 60 s roi chay lai (tu resume)"; sleep 60; }
  done
  return 1
}

train_done(){ # $1 = seed: 0 neu ca short va long da du 50 epoch
  python3 - "$OUT" "$1" <<'EOF'
import json, sys
from pathlib import Path
root, seed, ok = Path(sys.argv[1]), int(sys.argv[2]), True
for v in ('short', 'long'):
    try:
        st = json.loads((root / v / ('seed_%d' % seed) / 'run' / 'training_status.json').read_text())
        ok = ok and st.get('complete') is True
    except Exception:
        ok = False
sys.exit(0 if ok else 1)
EOF
}

backup(){ # $1 = nhan giai doan
  log "Backup $1"
  cd "$W" || return
  tar -cf rq2_LITE.tar --warning=no-file-changed --exclude='*.pth' --exclude='*.zip' \
      rq2_lstm_checked rq2_prep.log rq2_run.log rq2_auto.log rq2_auto_state.txt 2>/dev/null
  sha256sum rq2_LITE.tar > rq2_SHA256.txt
  if [ "$2" = "ckpt" ]; then
    find rq2_lstm_checked -name 'best_*.pth' -print0 | tar -cf rq2_BEST_CKPT.tar --null -T -
    sha256sum rq2_BEST_CKPT.tar >> rq2_SHA256.txt
  fi
  ls -lh rq2_LITE.tar rq2_BEST_CKPT.tar 2>/dev/null; cat rq2_SHA256.txt
}

# ---------- 1. Cho prep (tai data + preflight) ----------
P=$(pgrep -o -f "^bash $W/rq2_prep\.sh")
if [ -n "$P" ]; then
  log "1/6 Prep dang chay (PID $P) - hien log prep truc tiep, tu sang buoc 2 khi xong"
  tail -n +1 -f --pid="$P" "$W/rq2_prep.log"
else
  log "1/6 Prep khong con chay - 40 dong cuoi rq2_prep.log"
  tail -n 40 "$W/rq2_prep.log" 2>/dev/null
fi
grep -q PREP_XONG "$W/rq2_prep.log" 2>/dev/null || die "prep chua xong hoac loi (khong thay PREP_XONG)"
grep -q "preflight ma thoat 0" "$W/rq2_prep.log" || die "preflight loi (ma thoat khac 0) - xem log prep o tren"

# ---------- 2. Kiem SMOKE ----------
log "2/6 Kiem SMOKE short + long seed 42"
python3 - "$OUT" <<'EOF' || die "SMOKE khong dat - khong train"
import json, sys
from pathlib import Path
root, ok, tot = Path(sys.argv[1]), True, 0.0
for v in ('short', 'long'):
    p = root / v / 'seed_42' / 'smoke_report.json'
    try:
        s = json.loads(p.read_text())
    except Exception as e:
        print(v, 'KHONG DOC DUOC', p, e); ok = False; continue
    peak = s.get('peak_memory_gb_max_across_gpus', float('nan'))
    h = s.get('rough_train_only_50_epoch_hours', float('nan'))
    print('%-5s passed=%s mini_val=%s | %s | peak VRAM %.2f GB | %.3f s/microstep | %s microstep/epoch | train 50 epoch ~%.1f gio'
          % (v, s.get('passed'), s.get('mini_validation_passed'), s.get('gpu'), peak,
             s.get('seconds_per_microstep', float('nan')), s.get('train_microsteps_per_epoch_per_rank'), h))
    if s.get('passed') is not True or s.get('mini_validation_passed') is not True:
        ok = False
    if peak > 15.5:
        print('   CANH BAO: VRAM cao (> 15.5 GB) - co the OOM giua chung')
    tot += h
print('UOC TINH (+30%% cho validation): pilot 2 run ~%.0f gio | full them 4 run + test ~%.0f gio' % (tot * 1.3, tot * 2.6 + 3))
sys.exit(0 if ok else 1)
EOF

# ---------- 3. Pilot seed 42, chi validation (test van khoa) ----------
if train_done 42; then
  log "3/6 Pilot seed 42 da xong 50 epoch ca 2 nhanh - bo qua"
else
  log "3/6 PILOT seed 42: short roi long, 50 epoch moi nhanh, CHI validation"
  run_q "PILOT" --seeds 42 --train --validation-only --total-hours "$PILOT_H" || die "pilot loi 3 lan"
  train_done 42 || die "pilot chua du 50 epoch"
fi
python3 "$W/rq2_status.py" "$OUT"
backup "sau pilot"

# ---------- 4. Quy tac quyet dinh (chot truoc) ----------
log "4/6 QUYET DINH: long - short tai epoch tot nhat (val mAUC@), seed 42"
python3 - "$OUT" <<'EOF'
import json, re, sys
from pathlib import Path
root, best = Path(sys.argv[1]), {}
for v in ('short', 'long'):
    run = root / v / 'seed_42' / 'run'
    h = {}
    for line in (run / 'validation_history.jsonl').read_text().splitlines():
        if line.strip():
            d = json.loads(line); h[int(d['epoch'])] = d['metrics']
    st = json.loads((run / 'training_status.json').read_text())
    m = re.search(r'epoch_(\d+)', st.get('best_checkpoint', ''))
    e = int(m.group(1)) if m and int(m.group(1)) in h else max(h, key=lambda k: h[k].get('mAUC@', -1))
    best[v] = h[e]
    print('%-5s best epoch %2d: mAUC@ %.4f | AP@1.5s %.4f | mAP@ %.4f' % (v, e, h[e]['mAUC@'], h[e]['AP@1.5s'], h[e].get('mAP@', float('nan'))))
dauc = best['long']['mAUC@'] - best['short']['mAUC@']
dap = best['long']['AP@1.5s'] - best['short']['AP@1.5s']
print('long - short: mAUC@ %+.4f | AP@1.5s %+.4f  (nguong: CA HAI >= +0.0100)' % (dauc, dap))
ok = dauc >= 0.01 and dap >= 0.01
print('KET LUAN:', 'DAT -> chay du 3 seed' if ok else 'KHONG DAT -> mo test seed 42 de co so bao cao, roi dung')
sys.exit(0 if ok else 3)
EOF
DEC=$?
[ "$DEC" -eq 0 ] || [ "$DEC" -eq 3 ] || die "khong tinh duoc quy tac quyet dinh"

# ---------- 5. Chay tiep theo quyet dinh ----------
if [ "$DEC" -eq 0 ] && [ "$FULL" = "1" ]; then
  log "5/6 FULL: mo test seed 42 -> train + test seed 43, 44 cho ca 2 nhanh"
  run_q "FULL" --train --total-hours "$FULL_H" || die "full loi 3 lan"
elif [ "$DEC" -eq 0 ]; then
  log "5/6 DAT nhung FULL=0 -> dung. Muon chay tiep: FULL=1 voi cung lenh"
else
  log "5/6 KHONG DAT -> mo test seed 42 (short + long), khong train them"
  run_q "TEST seed 42" --seeds 42 --train --total-hours 10 || die "test seed 42 loi 3 lan"
fi

# ---------- 6. Tong ket + backup ----------
log "6/6 TONG KET"
python3 "$W/rq2_status.py" "$OUT"
backup "cuoi cung" ckpt
df -h "$W" | tail -1
log "XONG_HET - tai rq2_LITE.tar + rq2_BEST_CKPT.tar + rq2_SHA256.txt, kiem hash roi moi Destroy"
