# Third-party provenance

- **RiskProp:** https://github.com/xingyueye5/RiskProp, recorded source commit `579376fa1d879f28a9d68c2565d1e449de24b666`. The packages wrap/modify source preprocessing, runtime safety, bounded LSTM heads, capture and reporting. The pinned source ZIP retains the OpenMMLab Apache2.0 LICENSE; this observation does not substitute for auditing every upstream component.
- **TOP:** https://github.com/happytianhao/TOP. The pinned source ZIP root identifies commit `772c6e57e664ecde83f9d92d0641e1cba99abfb7` and retains the OpenMMLab Apache2.0 LICENSE. This project’s V2 implementation uses its own executed recipe and shared scalar evaluation wrapper.
- **AdaLEA:** Suzuki et al., CVPR2018. This is the team’s PyTorch reimplementation, not an official AdaLEA repository. See `rq1_reference/adalea_v4/code/provenance.json` for the paper and DSA/QRNN references, and the preserved `LICENSE_OpenMMLab` for reused components.
- **Nexar metadata:** preserved split/evaluation metadata retains its upstream LICENSE. No video/frame/feature dataset is included.

Original notices are retained under `licenses/` and in pinned source archives. No blanket license for the entire research release is invented here. Manuscript bibliography entries are outside the scope of this build and were not modified.
