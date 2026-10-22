"""Phase 40 — 100% air-gapped offline audit + launcher/demo-manifest checks."""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BACKEND = REPO_ROOT / "backend"
FRONTEND = REPO_ROOT / "frontend"

# External hosts that must never appear in source (allow 127.0.0.1/localhost only).
BANNED_PATTERNS = [
    re.compile(r"https?://(?!127\.0\.0\.1|localhost)[\w.-]+", re.IGNORECASE),
    re.compile(r"fonts\.googleapis\.com", re.IGNORECASE),
    re.compile(r"fonts\.gstatic\.com", re.IGNORECASE),
    re.compile(r"cdn\.jsdelivr\.net", re.IGNORECASE),
    re.compile(r"unpkg\.com", re.IGNORECASE),
]

SCAN_EXTENSIONS = {".py", ".ts", ".tsx", ".html", ".css"}


def _scan_files():
    for base in (BACKEND / "app", FRONTEND / "src", FRONTEND / "index.html", FRONTEND / "vite.config.ts"):
        paths = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in paths:
            if path.is_dir() or path.suffix not in SCAN_EXTENSIONS:
                continue
            yield path


def test_no_external_network_references_in_source():
    violations: list[str] = []
    for path in _scan_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in BANNED_PATTERNS:
            for match in pattern.finditer(text):
                snippet = match.group(0)
                # Allow documentation comments mentioning CDN names as warnings.
                if "offline" in text[max(0, match.start() - 120):match.start()].lower():
                    continue
                violations.append(f"{path.relative_to(REPO_ROOT)}: {snippet}")
    assert not violations, "external network refs found:\n" + "\n".join(violations[:20])


def test_frontend_bundles_fonts_locally():
    fonts_dir = FRONTEND / "public" / "fonts"
    css_files = list((FRONTEND / "src").rglob("*.css"))
    assert fonts_dir.is_dir(), "frontend/public/fonts/ must bundle .woff2 fonts offline"
    woff2 = list(fonts_dir.glob("*.woff2"))
    assert woff2, "no .woff2 font bundles found in frontend/public/fonts/"
    assert css_files, "no CSS files found under frontend/src"


def test_launchers_exist_and_bind_loopback_only():
    for launcher in ("start-windows.bat", "start-windows.ps1", "start-unix.sh"):
        for base in (REPO_ROOT / launcher, REPO_ROOT / "scripts" / launcher):
            assert base.is_file(), f"missing launcher {base}"
    for launcher in ("start-windows.bat", "start-windows.ps1", "start-unix.sh"):
        text = (REPO_ROOT / "scripts" / launcher).read_text(encoding="utf-8", errors="ignore")
        assert "30211" in text and "30222" in text, f"{launcher} must manage both ports"
        assert "127.0.0.1" in text or "localhost" in text, f"{launcher} must bind loopback"
    # Root launchers must delegate to scripts/ (single source of truth).
    root_bat = (REPO_ROOT / "start-windows.bat").read_text(encoding="utf-8", errors="ignore")
    assert "scripts" in root_bat


def test_demo_generator_manifest_contract(tmp_path):
    from scripts.generate_all_20_demos import generate_all_demo_artifacts

    out = tmp_path / "demos"
    result = generate_all_demo_artifacts(out)
    assert result["demo_count"] >= 20, f"expected 20+ demos, got {result['demo_count']}"
    manifest = json.loads((out / "demo_manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "1.0"
    families = {d["family"] for d in manifest["demos"]}
    for required in ("logistic", "knn", "svm", "tree", "forest", "xgboost",
                     "lightgbm", "catboost", "mlp", "pipeline"):
        assert required in families, f"demo family missing: {required}"
    assert (out / "demo_binary.csv").is_file()
    categories = {d["category"] for d in manifest["demos"]}
    assert len(categories) >= 5, f"expected 5+ gallery categories, got {categories}"
