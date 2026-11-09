"""One immutable training arm; legacy profiles are available for reference tests only."""
import os
ARM = "medium_h7"
HISTORY_MAP = {"short": 1, "long": 26, "medium_h7": 7, "medium_h13": 13}
VARIANTS = (ARM,)
VARIANT = os.environ.get("RQ2_VARIANT", ARM)
if VARIANT not in VARIANTS:
    raise ValueError("This ZIP trains only %s; unset a stale RQ2_VARIANT environment variable." % ARM)

def protocol(variant):
    if variant not in HISTORY_MAP:
        raise ValueError("Unknown history variant.")
    if variant in ("short", "long"):
        return "rq2-lstm-%s-nominalclock-b2-eff16-amp-cp-v2-checked" % variant
    return "rq2-lstm-%s-nominalclock-b2-eff16-amp-cp-medium-v1-20261029" % variant

METRIC_SCHEMA = "rq2-28-v1"
REQUIRED_TEST_METRICS = (
    "mAP","mAUC_grouped","mPAUC_sklearn_standardized","mPAUC_raw_divided_by_0.1",
    "mTTA_test_author_index10_inferred_s","maximum_actual_negative_frame_FAR",
    *("%s@%.1fs"%(metric,lead) for lead in (.5,1.,1.5)
      for metric in ("AP","AUC_full","pAUC_raw_div_0.1","mTTA_author_index10_inferred")),
    *("%s/%s"%(usage,metric) for usage in ("Public","Private")
      for metric in ("mAP","mAUC_grouped","mPAUC_sklearn_standardized","mPAUC_raw_divided_by_0.1")),
    "mTTA_Public_author_index10_inferred_s","mTTA_Private_author_index10_inferred_s",
)

def validate_metrics(values):
    import math
    if set(values) != set(REQUIRED_TEST_METRICS):
        raise ValueError("Require all28 metrics, including mTTA and each lead time.")
    for name,value in values.items():
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid metric: " + name)
        if not name.startswith("mTTA") and value > 1:
            raise ValueError("Ranking/rate metric outside[0,1]: " + name)

PROTOCOL = protocol(VARIANT)

def scientific_profile(variant=None):
    selected = VARIANT if variant is None else variant
    if selected not in HISTORY_MAP:
        raise ValueError("Unknown history variant.")
    history = HISTORY_MAP[selected]
    return dict(variant=selected, history_tokens=history,
                warmup_tokens=25, target_tokens=30, input_tokens=55,
                snippet_frames=5, native_frame_stride=3,
                maximum_native_frame_lookback=12+3*(history-1),
                nominal_frame_slots=history+4,
                clock="V6 nominal10Hz; not uniform physical seconds",
                recurrent="unidirectional single-layer LSTM2048, reset each bounded window",
                activation_checkpoint="residual blocks, nonreentrant, BN buffers restored on recompute",
                loss_target_grid="original30 endpoints; warm-up excluded")
