# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import importlib
from pathlib import Path

from opendde.config.model_registry import DEFAULT_MODEL_NAME

DEFAULT_DEPENDENCY_ROOT = "https://huggingface.co/aurekaresearch/OpenDDE/resolve/main"
DEFAULT_COMMON_ROOT = f"{DEFAULT_DEPENDENCY_ROOT}/common"


def _reload_dependency_url(monkeypatch, dependency_root, common_root=None):
    import opendde.config.dependency_url as dependency_url_module

    if dependency_root is None:
        monkeypatch.delenv("OPENDDE_DEPENDENCY_URL", raising=False)
    else:
        monkeypatch.setenv("OPENDDE_DEPENDENCY_URL", dependency_root)

    if common_root is None:
        monkeypatch.delenv("OPENDDE_COMMON_URL", raising=False)
    else:
        monkeypatch.setenv("OPENDDE_COMMON_URL", common_root)

    return importlib.reload(dependency_url_module)


def test_dependency_urls_default_to_public_https(monkeypatch):
    dependency_url_module = _reload_dependency_url(monkeypatch, None)

    assert dependency_url_module.URL["ccd_components_file"] == (
        f"{DEFAULT_COMMON_ROOT}/components.cif"
    )
    assert dependency_url_module.URL["ccd_components_rdkit_mol_file"] == (
        f"{DEFAULT_COMMON_ROOT}/components.cif.rdkit_mol.pkl"
    )
    assert dependency_url_module.URL[DEFAULT_MODEL_NAME] == (
        f"{DEFAULT_DEPENDENCY_ROOT}/opendde.pt"
    )
    assert dependency_url_module.dependency_url("opendde_abag.pt") == (
        f"{DEFAULT_DEPENDENCY_ROOT}/opendde_abag.pt"
    )
    assert dependency_url_module.CHECKPOINT_FILES[DEFAULT_MODEL_NAME] == "opendde.pt"


def test_default_checkpoint_path_uses_released_filename(tmp_path):
    from opendde.utils.download import resolve_checkpoint_path

    class Config(dict):
        def __getattr__(self, key):
            return self[key]

    cfg = Config(
        load_checkpoint_path="",
        load_checkpoint_dir=str(tmp_path),
        model_name=DEFAULT_MODEL_NAME,
    )

    assert resolve_checkpoint_path(cfg) == str(tmp_path / "opendde.pt")


def test_dependency_url_supports_custom_https_root(monkeypatch):
    dependency_url_module = _reload_dependency_url(
        monkeypatch, "https://example.com/opendde/dependency/"
    )

    assert dependency_url_module.dependency_url("opendde.pt") == (
        "https://example.com/opendde/dependency/opendde.pt"
    )
    assert dependency_url_module.common_url("components.cif") == (
        "https://example.com/opendde/dependency/components.cif"
    )


def test_common_url_supports_independent_custom_root(monkeypatch):
    dependency_url_module = _reload_dependency_url(
        monkeypatch,
        "https://example.com/opendde/dependency/",
        "https://example.com/opendde/common/",
    )

    assert dependency_url_module.URL[DEFAULT_MODEL_NAME] == (
        "https://example.com/opendde/dependency/opendde.pt"
    )
    assert dependency_url_module.URL["ccd_components_file"] == (
        "https://example.com/opendde/common/components.cif"
    )


ALPHAFOLD_DB_ROOT = "https://storage.googleapis.com/alphafold-databases/v3.0"

_SEARCH_DB_ENV_VARS = (
    "OPENDDE_SEARCH_DATABASE_URL",
    "OPENDDE_PDB_SEQRES_URL",
    "OPENDDE_RFAM_DB_URL",
    "OPENDDE_NT_RNA_DB_URL",
    "OPENDDE_RNACENTRAL_DB_URL",
)


def _reload_with_search_db_env(monkeypatch, env):
    import opendde.config.dependency_url as dependency_url_module

    for key in _SEARCH_DB_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(dependency_url_module)


def test_search_database_urls_default_to_alphafold_v3_archives(monkeypatch):
    module = _reload_with_search_db_env(monkeypatch, {})

    assert module.SEARCH_DATABASE_URL == {
        "pdb_seqres": f"{ALPHAFOLD_DB_ROOT}/pdb_seqres_2022_09_28.fasta.zst",
        "rfam": (
            f"{ALPHAFOLD_DB_ROOT}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst"
        ),
        "nt_rna": (
            f"{ALPHAFOLD_DB_ROOT}/"
            "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst"
        ),
        "rnacentral": (
            f"{ALPHAFOLD_DB_ROOT}/rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst"
        ),
    }


def test_search_database_root_override_applies_to_all(monkeypatch):
    module = _reload_with_search_db_env(
        monkeypatch, {"OPENDDE_SEARCH_DATABASE_URL": "https://mirror.example.com/db/"}
    )

    assert module.SEARCH_DATABASE_URL["rfam"] == (
        "https://mirror.example.com/db/"
        "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst"
    )


def test_search_database_per_database_override_takes_priority(monkeypatch):
    module = _reload_with_search_db_env(
        monkeypatch,
        {
            "OPENDDE_SEARCH_DATABASE_URL": "https://mirror.example.com/db",
            "OPENDDE_PDB_SEQRES_URL": "https://my-s3.example.com/pdb_seqres.fasta",
        },
    )

    assert module.SEARCH_DATABASE_URL["pdb_seqres"] == (
        "https://my-s3.example.com/pdb_seqres.fasta"
    )
    assert module.SEARCH_DATABASE_URL["nt_rna"].startswith(
        "https://mirror.example.com/db/"
    )


def test_download_from_url_uses_urlretrieve(monkeypatch, tmp_path):
    import opendde.utils.download as download_module

    calls = []

    def fake_urlretrieve(url, filename, reporthook=None):
        calls.append((url, filename, reporthook is not None))
        Path(filename).write_bytes(b"demo")

    monkeypatch.setattr(download_module.urllib.request, "urlretrieve", fake_urlretrieve)

    output_path = tmp_path / "components.cif"
    download_module.download_from_url(
        "https://example.com/components.cif",
        str(output_path),
        check_weight=False,
    )

    assert output_path.read_bytes() == b"demo"
    assert calls == [("https://example.com/components.cif", str(output_path), True)]


def test_download_from_url_decompresses_zst_to_requested_path(monkeypatch, tmp_path):
    import opendde.utils.download as download_module

    url = "https://example.com/db.fasta.zst"
    calls = []

    def fake_urlretrieve(url, filename, reporthook=None):
        calls.append((url, filename, reporthook is not None))
        Path(filename).write_bytes(b"compressed")

    def fake_decompress_zst(zst_path, output_path, source_url):
        assert Path(zst_path).read_bytes() == b"compressed"
        assert source_url == url
        Path(output_path).write_text(">seq\nACGU\n")

    monkeypatch.setattr(download_module.urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(download_module, "_decompress_zst", fake_decompress_zst)

    output_path = tmp_path / "db.fasta"
    download_module.download_from_url(url, str(output_path), check_weight=False)

    assert output_path.read_text() == ">seq\nACGU\n"
    assert len(calls) == 1
    assert calls[0][0] == url
    assert calls[0][2] is True
    assert calls[0][1].endswith(".zst")
