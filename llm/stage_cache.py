import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


CACHE_SCHEMA_VERSION = 1


def _canonical_digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StageCache:

    def __init__(self, video_path, identity, cache_dir=None):

        self.video_fingerprint = self._fingerprint(video_path)
        self.identity_digest = _canonical_digest(identity)
        root = cache_dir or os.getenv(
            "GEMINI_CACHE_DIR",
            "media/output/.gemini_cache",
        )
        self.root = Path(root)
        self.path = self.root / f"{self.video_fingerprint}.json"
        self._secure_root()
        self.data = self._load()

    def _secure_root(self):

        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass

    def _fingerprint(self, video_path):

        digest = hashlib.sha256()
        with open(video_path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _empty_data(self):

        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "video_fingerprint": self.video_fingerprint,
            "identity_digest": self.identity_digest,
            "stages": {},
            "upload": None,
        }

    def _load(self):

        try:
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
            with open(self.path, "r", encoding="utf-8") as source:
                cached = json.load(source)
        except (OSError, ValueError, TypeError):
            return self._empty_data()

        if (
            not isinstance(cached, dict)
            or cached.get("schema_version") != CACHE_SCHEMA_VERSION
            or cached.get("video_fingerprint") != self.video_fingerprint
        ):
            return self._empty_data()

        upload = cached.get("upload")
        if not isinstance(upload, dict):
            upload = None

        if cached.get("identity_digest") != self.identity_digest:
            # A changed model or prompt invalidates all analysis stages. The
            # provider upload still represents the same fingerprinted video.
            reset = self._empty_data()
            reset["upload"] = upload
            return reset

        stages = cached.get("stages")
        if not isinstance(stages, dict):
            stages = {}

        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "video_fingerprint": self.video_fingerprint,
            "identity_digest": self.identity_digest,
            "stages": stages,
            "upload": upload,
        }

    def clear_stage(self, name):

        self.data["stages"].pop(name, None)
        self._save()

    def get_stage(self, name, context):

        record = self.data["stages"].get(name)
        if not isinstance(record, dict):
            return None

        try:
            if record.get("context_digest") != _canonical_digest(context):
                return None
            value = record.get("value")
            json.dumps(value, allow_nan=False)
            return value
        except (TypeError, ValueError):
            return None

    def set_stage(self, name, context, value):

        encoded = json.dumps(value, allow_nan=False)
        self.data["stages"][name] = {
            "context_digest": _canonical_digest(context),
            "value": json.loads(encoded),
        }
        self._save()

    def get_upload(self):

        upload = self.data.get("upload")
        if not isinstance(upload, dict):
            return None
        if not all(
            isinstance(upload.get(key), str) and upload[key]
            for key in ("name", "uri", "mime_type")
        ):
            return None
        try:
            saved_at = float(upload.get("saved_at"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(saved_at) or saved_at < 0:
            return None
        return {
            "name": upload["name"],
            "uri": upload["uri"],
            "mime_type": upload["mime_type"],
            "saved_at": saved_at,
        }

    def set_upload(self, upload):

        allowed = {
            key: upload[key]
            for key in ("name", "uri", "mime_type", "saved_at")
            if key in upload
        }
        if not all(
            isinstance(allowed.get(key), str) and allowed[key]
            for key in ("name", "uri", "mime_type")
        ):
            raise ValueError("Gemini upload metadata is incomplete.")
        allowed["saved_at"] = float(allowed.get("saved_at", 0.0))
        self.data["upload"] = allowed
        self._save()

    def clear_upload(self):

        self.data["upload"] = None
        self._save()

    def _save(self):

        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass

        payload = json.dumps(
            self.data,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        fd, temporary_path = tempfile.mkstemp(
            prefix=".gemini-stage-",
            suffix=".tmp",
            dir=self.root,
        )
        try:
            try:
                os.fchmod(fd, 0o600)
            except OSError:
                pass
            with os.fdopen(fd, "wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_path, self.path)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        finally:
            try:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
            except OSError:
                pass
