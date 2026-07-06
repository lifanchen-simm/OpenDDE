# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from pathlib import Path


def test_template_search_downloads_missing_database(monkeypatch, tmp_path):
    from runner import template_search

    calls = []

    def fake_download(url, path, check_weight=True):
        calls.append((url, path, check_weight))
        Path(path).write_text(">seq\nACDE\n")

    monkeypatch.setattr(template_search, "download_from_url", fake_download)
    monkeypatch.setattr(template_search, "run_hmmsearch_with_a3m", lambda **_: "")

    template_search.run_template_search(
        msa_for_template_search_dir=str(tmp_path),
        msa_for_template_search_name="missing",
        hmmsearch_binary_path="/bin/echo",
        hmmbuild_binary_path="/bin/echo",
        seqres_database_path=str(tmp_path / "db" / "pdb_seqres.fasta"),
    )

    assert calls == [
        (
            template_search.TEMPLATE_SEARCH_DATABASE_URL,
            str(tmp_path / "db" / "pdb_seqres.fasta"),
            False,
        )
    ]
    assert (tmp_path / "hmmsearch.a3m").exists()


def test_rna_msa_search_downloads_missing_databases(monkeypatch, tmp_path):
    from runner import rna_msa_search

    calls = []

    def fake_download(url, path, check_weight=True):
        calls.append((url, path, check_weight))
        Path(path).write_text(">seq\nACGU\n")

    class DummyMsa:
        def to_a3m(self):
            return ">query\nACGU\n"

    monkeypatch.setattr(rna_msa_search, "download_from_url", fake_download)
    monkeypatch.setattr(rna_msa_search, "_get_rna_msa", lambda **_: DummyMsa())

    db_paths = {
        "ntrna_database_path": tmp_path / "db" / "nt.fasta",
        "rfam_database_path": tmp_path / "db" / "rfam.fasta",
        "rna_central_database_path": tmp_path / "db" / "rnacentral.fasta",
    }

    rna_msa_search.run_rna_msa_search(
        rna_seq_for_msa_search="ACGU",
        rna_result_path=str(tmp_path / "out"),
        rna_seq_id="rna",
        nhmmer_binary_path="/bin/echo",
        hmmalign_binary_path="/bin/echo",
        hmmbuild_binary_path="/bin/echo",
        ntrna_database_path=str(db_paths["ntrna_database_path"]),
        rfam_database_path=str(db_paths["rfam_database_path"]),
        rna_central_database_path=str(db_paths["rna_central_database_path"]),
    )

    assert calls == [
        (
            rna_msa_search.NT_SEARCH_DATABASE_URL,
            str(db_paths["ntrna_database_path"]),
            False,
        ),
        (
            rna_msa_search.RFAM_SEARCH_DATABASE_URL,
            str(db_paths["rfam_database_path"]),
            False,
        ),
        (
            rna_msa_search.RNACENTRAL_SEARCH_DATABASE_URL,
            str(db_paths["rna_central_database_path"]),
            False,
        ),
    ]
    assert (tmp_path / "out" / "rna" / "rna_msa.a3m").read_text() == ">query\nACGU\n"
