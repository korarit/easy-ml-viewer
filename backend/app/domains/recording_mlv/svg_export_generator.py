"""Phase 48 — import shim (canonical: ``export_reporting.svg_export_service``).

The phase-48 test plan addresses this module as
``app.domains.recording.svg_export_generator``; the DDD home for exports is
``export_reporting``, so this file re-exports that implementation for both
import styles.
"""

from app.domains.export_reporting.svg_export_service import SvgExportGenerator

__all__ = ["SvgExportGenerator"]
