# Memory Index

- [SSPM-Net complex-RI work state](sspm-net-complex-ri-state.md) — branch feature/complex-ri-merlin, winning config, thesis findings, open items
- [Work on branches](work-on-branches.md) — experimental work goes to feature/* branches, never directly to main; no Claude trailers in commits
- [Colab checkpoint commits](colab-checkpoint-commits.md) — commit/push code + session context to the feature branch at regular intervals; the VM is ephemeral
- [Phase decoding fix](sspm-net-complex-ri-state.md) — the uint8 pha TIFFs are FULL-RANGE [0,2pi) SLC phase (pha/255*2pi), not folded; complex SLC is recoverable
