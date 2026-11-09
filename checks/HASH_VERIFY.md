# Hash audit

See [artifact_verification.json](artifact_verification.json) for complete measured file hashes and sidecar checks, and [runtime_source_verification.json](runtime_source_verification.json) for55 runtime source hash comparisons.

- V3.2 bundle and both summary/log archives match the five full SHA256 values reported in the handover.
- H7 executed ZIP checksum matches the original v1 archive from Downloads, not the later FIXED archive. See [bundle comparison](h7_code_bundle_comparison.json).
- The executed earlier V6 worker matches its runtime contract. Package-vs-runtime is a separate comparison; no manuscript statement was deleted.
- All nine inspected evaluator module comparisons (three modules×three RQ2 packages) match the V3.2 canonical source.
- The seed44 rerun core files do not uniquely distinguish wrapper ZIP variants.

Original archives are not edited. File-size filtering, lossless gzip and any security redactions are separately recorded. A checksum is a byte-integrity check, not evidence that the scientific method is correct.
