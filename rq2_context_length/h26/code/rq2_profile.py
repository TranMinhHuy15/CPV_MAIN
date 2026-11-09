"""Immutable scientific arms; supervisor selects the arm before starting a process."""
import os

VARIANTS = ('short', 'long')
VARIANT = os.environ.get('RQ2_VARIANT', 'short')


def protocol(variant):
    if variant not in VARIANTS:
        raise ValueError('RQ2_VARIANT must be short or long.')
    return 'rq2-lstm-%s-nominalclock-b2-eff16-amp-cp-v2-checked' % variant


METRIC_SCHEMA = 'rq2-28-v1'
REQUIRED_TEST_METRICS = (
    'mAP','mAUC_grouped','mPAUC_sklearn_standardized','mPAUC_raw_divided_by_0.1',
    'mTTA_test_author_index10_inferred_s','maximum_actual_negative_frame_FAR',
    *('%s@%.1fs'%(metric,lead) for lead in (.5,1.,1.5)
      for metric in ('AP','AUC_full','pAUC_raw_div_0.1','mTTA_author_index10_inferred')),
    *('%s/%s'%(usage,metric) for usage in ('Public','Private')
      for metric in ('mAP','mAUC_grouped','mPAUC_sklearn_standardized','mPAUC_raw_divided_by_0.1')),
    'mTTA_Public_author_index10_inferred_s','mTTA_Private_author_index10_inferred_s',
)


def validate_metrics(values):
    import math
    if set(values) != set(REQUIRED_TEST_METRICS):
        raise ValueError('Require the complete fixed28-metric schema, including mTTA and all lead times.')
    for name,value in values.items():
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid metric: '+name)
        if not name.startswith('mTTA') and value>1:
            raise ValueError('Probability/ranking metric is outside[0,1]: '+name)


PROTOCOL = protocol(VARIANT)


def scientific_profile(variant=VARIANT):
    return dict(variant=variant, history_tokens=1 if variant == 'short' else 26,
                warmup_tokens=25, target_tokens=30, input_tokens=55,
                snippet_frames=5, native_frame_stride=3,
                maximum_native_frame_lookback=12 if variant == 'short' else 87,
                nominal_frame_slots=5 if variant == 'short' else 30,
                clock='V6 nominal10Hz; not uniform physical seconds',
                recurrent='unidirectional single-layer LSTM2048, reset each bounded window',
                activation_checkpoint='residual blocks, nonreentrant, BN buffers restored on recompute',
                loss_target_grid='original30 endpoints; warm-up excluded')
