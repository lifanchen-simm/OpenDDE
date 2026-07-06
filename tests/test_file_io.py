# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import gzip
import pickle

from opendde.utils.file_io import LMDBDict, load_gzip_pickle


def test_load_gzip_pickle_roundtrips_lmdbdict(tmp_path):
    """load_gzip_pickle restores a gzip-pickled object as-is.

    The legacy package-name remapping shim was removed in the inference-only
    build, so this only exercises a plain round-trip.
    """
    pkl_path = tmp_path / "data.pkl.gz"
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(LMDBDict("some.lmdb"), f)

    loaded = load_gzip_pickle(pkl_path)

    assert isinstance(loaded, LMDBDict)
    assert loaded.path == "some.lmdb"
