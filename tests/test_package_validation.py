"""Untrusted portable packages must fail validation before reading outside their root."""

import json

import pytest
from bundle_helpers import synthetic_spec

from iceclaw.reports import package as pkg


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "package"
    pkg.write_package(synthetic_spec(tmp_path / "inputs"), root)
    return root


def edit_manifest(root, change):
    path = root / pkg.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    change(manifest)
    body = {k: v for k, v in manifest.items() if k != "package_sha256"}
    manifest["package_sha256"] = pkg.sha256_bytes(pkg.canonical(body).encode())
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("name", ["../outside.tif", "/outside.tif", r"..\outside.tif", "C:/x"])
def test_source_paths_cannot_escape_package(source, name):
    edit_manifest(source, lambda m: m["files"][0].update(file=name))
    with pytest.raises(pkg.PackageError, match="INVALID_PATH"):
        pkg.load_package(source)


def test_source_symlink_cannot_escape_package(source, tmp_path):
    manifest = pkg.read_manifest(source)
    path = source / manifest["files"][0]["file"]
    outside = tmp_path / "outside"
    path.rename(outside)
    path.symlink_to(outside)
    # Exact same bytes and hash: containment must be checked independently of integrity.
    with pytest.raises(pkg.PackageError, match="INVALID_PATH"):
        pkg.load_package(source)


def test_manifest_symlink_cannot_escape_package(source, tmp_path):
    path = source / pkg.MANIFEST_NAME
    outside = tmp_path / "manifest.json"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(pkg.PackageError, match="INVALID_PATH"):
        pkg.read_manifest(source)


@pytest.mark.parametrize("field", ["file", "file_id", "modality"])
def test_duplicate_source_identity_is_rejected(source, field):
    edit_manifest(source, lambda m: m["files"][1].update({field: m["files"][0][field]}))
    with pytest.raises(pkg.PackageError, match="INVALID_MANIFEST.*duplicate"):
        pkg.load_package(source)


def test_declared_size_is_checked(source):
    edit_manifest(source, lambda m: m["files"][0].update(size=0))
    with pytest.raises(pkg.PackageError, match="CHECKSUM_MISMATCH"):
        pkg.load_package(source)


def test_unknown_source_kind_is_rejected(source):
    edit_manifest(source, lambda m: m["files"][0].update(kind="unknown"))
    with pytest.raises(pkg.PackageError, match="INVALID_MANIFEST.*unknown source kind"):
        pkg.load_package(source)


@pytest.mark.parametrize("value", [[], None, 42, "manifest"])
def test_nonobject_manifest_is_reported(source, value):
    (source / pkg.MANIFEST_NAME).write_text(json.dumps(value))
    with pytest.raises(pkg.PackageError, match="INVALID_MANIFEST"):
        pkg.load_package(source)


@pytest.mark.parametrize("field", ["geometry", "region", "files"])
def test_missing_manifest_section_is_reported(source, field):
    edit_manifest(source, lambda m: m.pop(field))
    with pytest.raises(pkg.PackageError, match="INVALID_MANIFEST"):
        pkg.load_package(source)
