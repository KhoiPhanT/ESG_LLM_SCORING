"""
Shared cache utilities for the ESG pipeline.

The manager is intentionally small: it centralizes fingerprints, atomic JSON
writes, force-rebuild flags, and a manifest that makes cache decisions visible.
It does not decide scoring behavior.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from typing import Any


class CacheManager:
    MANIFEST_PATH = "outputs/cache/cache_manifest.json"
    MANIFEST_SCHEMA = 1
    _SESSION_COUNTS: dict[tuple[str, str], dict[str, int]] = {}

    def __init__(self, run_key: str = "global", manifest_path: str | None = None):
        self.run_key = run_key or "global"
        self.manifest_path = manifest_path or self.MANIFEST_PATH
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        self._manifest = self._load_manifest()

    @staticmethod
    def hash_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def file_fingerprint(cls, path: str, extra: dict[str, Any] | None = None) -> str:
        stat = os.stat(path)
        payload = {
            "path": os.path.abspath(path),
            "size": stat.st_size,
            "mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)),
            "sha256": cls.hash_file(path),
        }
        if extra:
            payload["extra"] = extra
        return cls.hash_json(payload)

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    @staticmethod
    def hash_json(value: Any) -> str:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def folder_fingerprint(cls, paths: list[str], extra: dict[str, Any] | None = None) -> str:
        payload = {
            "files": [
                {
                    "path": os.path.abspath(path),
                    "fingerprint": cls.file_fingerprint(path),
                }
                for path in sorted(paths)
                if os.path.exists(path)
            ],
            "extra": extra or {},
        }
        return cls.hash_json(payload)

    @staticmethod
    def atomic_write_json(path: str, payload: Any, indent: int | None = 2) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=indent, default=str)
        os.replace(tmp_path, path)

    @staticmethod
    def load_json(path: str) -> Any | None:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @classmethod
    def forced_stages(cls) -> set[str]:
        raw = os.environ.get("ESG_FORCE_REBUILD", "")
        return {part.strip().lower() for part in raw.split(",") if part.strip()}

    @classmethod
    def is_forced(cls, stage: str) -> bool:
        forced = cls.forced_stages()
        stage = (stage or "").strip().lower()
        return "all" in forced or stage in forced

    def record(
        self,
        stage: str,
        status: str,
        schema_version: str | int,
        input_fingerprint: str,
        path: str | None = None,
        reason: str | None = None,
        error: str | None = None,
    ) -> None:
        manifest = self._manifest
        manifest.setdefault("schema_version", self.MANIFEST_SCHEMA)
        runs = manifest.setdefault("runs", {})
        run = runs.setdefault(self.run_key, {})
        count_key = (self.run_key, stage)
        counts = self._SESSION_COUNTS.setdefault(
            count_key,
            {"hit": 0, "rebuilt": 0, "failed": 0},
        )
        if status in counts:
            counts[status] += 1
        run[stage] = {
            "status": status,
            "schema_version": schema_version,
            "input_fingerprint": input_fingerprint,
            "path": path,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "reason": reason,
            "error": error,
            "session_counts": dict(counts),
        }
        self._save_manifest()

    def latest(self, stage: str) -> dict | None:
        return (
            self._manifest
            .get("runs", {})
            .get(self.run_key, {})
            .get(stage)
        )

    def print_summary(self) -> None:
        if os.environ.get("ESG_CACHE_STATUS", "0") != "1":
            return
        run = self._manifest.get("runs", {}).get(self.run_key, {})
        if not run:
            print("[CACHE] No cache records for this run")
            return
        for stage, record in run.items():
            status = record.get("status", "?")
            reason = record.get("reason") or ""
            suffix = f" because {reason}" if reason else ""
            counts = record.get("session_counts") or {}
            if counts:
                count_text = ", ".join(f"{key}={value}" for key, value in counts.items())
                print(f"[CACHE] {stage}: {status} ({count_text}){suffix}")
            else:
                print(f"[CACHE] {stage}: {status}{suffix}")

    def _load_manifest(self) -> dict:
        payload = self.load_json(self.manifest_path)
        if not isinstance(payload, dict) or payload.get("schema_version") != self.MANIFEST_SCHEMA:
            return {"schema_version": self.MANIFEST_SCHEMA, "runs": {}}
        return payload

    def _save_manifest(self) -> None:
        self.atomic_write_json(self.manifest_path, self._manifest, indent=2)
