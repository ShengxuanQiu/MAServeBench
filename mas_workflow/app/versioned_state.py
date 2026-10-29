"""Run-local versioned state with explicit, JSON-serializable snapshots."""
from __future__ import annotations

import json
import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StateVersion:
    state_object_id: str
    version: int
    value: Any
    producer_operation_id: str | None


class VersionedStateStore:
    """Keep immutable versions; restored snapshots have no live source writer.

    The store is scoped to one native run. Persistent services can supply a
    snapshot before execution; replay consumes the recorded values without
    calling that service. Transactions may hold ``lock`` across trace emission.
    """
    def __init__(self):
        self.lock = threading.RLock()
        self._objects: dict[str, list[StateVersion]] = {}

    @staticmethod
    def _key(key):
        if not isinstance(key, str) or not key:
            raise ValueError("state_object_id must be nonempty text")

    def read(self, key: str, version: int | None = None) -> StateVersion:
        self._key(key)
        with self.lock:
            values = self._objects[key]
            if version is None:
                item = values[-1]
            else:
                if type(version) is not int or not 0 <= version < len(values):
                    raise ValueError("Unknown state version")
                item = values[version]
            return StateVersion(key, item.version, deepcopy(item.value), item.producer_operation_id)

    def write(self, key: str, value: Any, producer_operation_id: str,
              expected_version: int | None = None) -> StateVersion:
        self._key(key)
        # Reject non-JSON values before changing state or recording provenance.
        json.dumps(value, allow_nan=False)
        if not isinstance(producer_operation_id, str) or not producer_operation_id:
            raise ValueError("A state write requires a producer operation ID")
        with self.lock:
            values = self._objects.get(key, [])
            current = len(values) - 1
            if expected_version is not None and (type(expected_version) is not int or expected_version != current):
                raise ValueError("State version changed before write")
            item = StateVersion(key, current + 1, deepcopy(value), producer_operation_id)
            self._objects.setdefault(key, []).append(item)
            return self.read(key)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {"schema_version": "masbench.state_snapshot/1.0.0", "objects": {
                key: [{"version": item.version, "value": deepcopy(item.value)} for item in values]
                for key, values in sorted(self._objects.items())}}

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore into an empty run; source operation IDs never cross runs."""
        if snapshot.get("schema_version") != "masbench.state_snapshot/1.0.0" or not isinstance(snapshot.get("objects"), dict):
            raise ValueError("Unsupported state snapshot")
        objects = {}
        for key, values in snapshot["objects"].items():
            self._key(key)
            if (not isinstance(values, list) or not values or
                    any(not isinstance(v, dict) or type(v.get("version")) is not int or v["version"] != i or "value" not in v
                        for i, v in enumerate(values))):
                raise ValueError("Snapshot versions must start at zero and be contiguous")
            json.dumps(values, allow_nan=False)
            objects[key] = [StateVersion(key, v["version"], deepcopy(v["value"]), None) for v in values]
        with self.lock:
            if self._objects:
                raise ValueError("Restore requires an empty state store")
            self._objects = objects
