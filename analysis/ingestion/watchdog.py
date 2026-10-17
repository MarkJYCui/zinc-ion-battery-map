"""
Autonomous Ingestion Watchdog for Battery Cycler & Potentiostat Data Exports.
Monitors an inbox directory for automated exports from laboratory cyclers,
ensures file write stabilization, and triggers autonomous analysis pipelines.
"""

import time
import shutil
import logging
from pathlib import Path
from typing import Callable, List, Optional, Set, Dict, Any, Union

from models import ElectrochemicalAnalysisReport

logger = logging.getLogger(__name__)


class CyclerWatchdog:
    """Monitors ingestion folders for incoming cycler export files."""

    SUPPORTED_EXTENSIONS = {".xlsx", ".dat", ".csv", ".mpr", ".txt", ".mpt"}

    def __init__(
        self,
        watch_dir: Union[str, Path],
        processed_dir: Optional[Union[str, Path]] = None,
        callback: Optional[Callable[[Path], Optional[ElectrochemicalAnalysisReport]]] = None,
        stabilization_wait_s: float = 0.5,
    ):
        self.watch_dir = Path(watch_dir)
        self.processed_dir = Path(processed_dir) if processed_dir else self.watch_dir / "processed"
        self.callback = callback
        self.stabilization_wait_s = stabilization_wait_s
        self.processed_files: Set[str] = set()

        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _is_file_stabilized(self, filepath: Path) -> bool:
        try:
            size1 = filepath.stat().st_size
            time.sleep(self.stabilization_wait_s)
            size2 = filepath.stat().st_size
            return size1 == size2 and size1 > 0
        except (FileNotFoundError, PermissionError):
            return False

    def scan_inbox(self) -> List[Path]:
        new_files: List[Path] = []
        if not self.watch_dir.exists():
            return []

        for p in self.watch_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            if p.name in self.processed_files:
                continue
            if p.name.startswith("."):
                continue
            if not self._is_file_stabilized(p):
                continue
            new_files.append(p)
        return new_files

    def process_file(self, filepath: Path) -> Optional[ElectrochemicalAnalysisReport]:
        logger.info(f"Watchdog detected new cycler file: {filepath.name}")
        report = None
        if self.callback:
            try:
                report = self.callback(filepath)
            except Exception as e:
                logger.error(f"Error processing {filepath.name}: {e}")
                raise e

        self.processed_files.add(filepath.name)
        dest = self.processed_dir / filepath.name
        if filepath.exists():
            try:
                shutil.move(str(filepath), str(dest))
            except Exception as e:
                logger.warning(f"Could not archive {filepath.name}: {e}")

        return report

    def poll_once(self) -> List[ElectrochemicalAnalysisReport]:
        reports: List[ElectrochemicalAnalysisReport] = []
        new_files = self.scan_inbox()
        for f in new_files:
            rep = self.process_file(f)
            if rep is not None:
                reports.append(rep)
        return reports

    def run_polling_loop(
        self,
        interval_s: float = 2.0,
        max_iterations: Optional[int] = None,
    ):
        logger.info(f"Starting CyclerWatchdog on: {self.watch_dir}")
        iterations = 0
        try:
            while max_iterations is None or iterations < max_iterations:
                self.poll_once()
                time.sleep(interval_s)
                iterations += 1
        except KeyboardInterrupt:
            logger.info("Watchdog monitoring stopped by user.")
