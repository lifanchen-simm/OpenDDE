# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import numpy as np
import torch
from biotite.structure import AtomArray

from opendde.data import utils as data_utils


def _make_atom_array() -> AtomArray:
    atom_array = AtomArray(2)
    atom_array.coord = np.zeros((2, 3), dtype=np.float32)
    atom_array.set_annotation("is_resolved", np.array([True, False]))
    return atom_array


def test_save_structure_cif_does_not_save_wounresol_by_default(monkeypatch, tmp_path):
    saved_paths = []

    def fake_save_atoms_to_cif(output_cif_file, atom_array, entity_poly_type, pdb_id):
        saved_paths.append(output_cif_file)

    monkeypatch.setattr(data_utils, "save_atoms_to_cif", fake_save_atoms_to_cif)

    output_path = str(tmp_path / "model.cif")
    data_utils.save_structure_cif(
        atom_array=_make_atom_array(),
        pred_coordinate=torch.zeros((2, 3)),
        output_fpath=output_path,
        entity_poly_type={},
        pdb_id="test",
    )

    assert saved_paths == [output_path]


def test_save_structure_cif_can_save_wounresol_when_requested(monkeypatch, tmp_path):
    saved = []

    def fake_save_atoms_to_cif(output_cif_file, atom_array, entity_poly_type, pdb_id):
        saved.append((output_cif_file, len(atom_array)))

    monkeypatch.setattr(data_utils, "save_atoms_to_cif", fake_save_atoms_to_cif)

    output_path = str(tmp_path / "model.cif")
    data_utils.save_structure_cif(
        atom_array=_make_atom_array(),
        pred_coordinate=torch.zeros((2, 3)),
        output_fpath=output_path,
        entity_poly_type={},
        pdb_id="test",
        save_wo_unresolved=True,
    )

    assert saved == [
        (output_path, 2),
        (str(tmp_path / "model_wounresol.cif"), 1),
    ]
