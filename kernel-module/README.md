# UVM driver

`nvidia-module/kernel-open/nvidia-uvm/` contains the complete modified UVM
source. The surrounding NVIDIA driver is obtained at build time from the
official NVIDIA source repository:

- Version: **595.71.05**
- Commit: `095de56d46690f857d7a5aa2512eadee71ed013f`
- Archive SHA-256: `836c03b5445e04975a54c2afc76cca7403e9abb4c13f71d02037bb0672692b21`

```bash
bash scripts/build_driver.sh
```

This downloads and verifies the upstream archive, overlays the AdaptMoE UVM
directory, and builds all modules in `.build/nvidia-595.71.05`. It does not
install or load them. Matching kernel headers, the compiler, and NVIDIA base
driver are required. To build against a particular header tree, pass
`SYSSRC=/path/to/headers` after the script name.

For offline preparation, set `NVIDIA_SOURCE_ARCHIVE` to a local copy of the
same archive. `DRIVER_BUILD_DIR` selects a different, new build directory;
set `CUSTOM_UVM` accordingly when running the model comparison.

The source tree includes both NVIDIA licenses in `nvidia-module/COPYING` and
the original per-file notices. The repository stores UVM sources as an overlay;
base-driver sources are supplied by the pinned upstream archive. The build receipt
records the upstream revision, archive checksum, and every overlaid UVM hash.
