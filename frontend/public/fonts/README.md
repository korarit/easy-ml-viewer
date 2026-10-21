# Bundled offline fonts (Phase 01)

Place the licensed font binaries here — NO external downloads at runtime:

- Inter-Regular.woff2
- Inter-Medium.woff2
- Inter-SemiBold.woff2
- JetBrainsMono-Regular.woff2
- JetBrainsMono-Medium.woff2

`studio.css` declares `@font-face` against `/fonts/*.woff2` with
system-font fallbacks, so the studio stays fully air-gapped even if a
file is temporarily missing.
