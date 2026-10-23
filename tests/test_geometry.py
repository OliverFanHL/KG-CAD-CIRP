"""Optional CAD-kernel integration on known solids, including STEP unit conversion."""
from __future__ import annotations

import pytest

pytest.importorskip("OCP")

from cadkg.geometry import extract_step


def export_shape(shape, path):
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    writer = STEPControl_Writer()
    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
    assert writer.Write(str(path)) == IFSelect_RetDone


def test_known_box_measurements(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    path = tmp_path / "box.stp"
    export_shape(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), path)
    result = extract_step(path)
    assert result["geometry_audit"]["valid_shape"]
    assert result["measurements"]["aabb_large_mm"]["value"] == pytest.approx(30, abs=1e-5)
    assert result["measurements"]["volume_mm3"]["value"] == pytest.approx(6000)
    assert result["measurements"]["area_mm2"]["value"] == pytest.approx(2200)
    assert result["measurements"]["aabb_large_mm"]["status"] == "computed"
    assert result["cylinder_candidates"] == []


def test_cylinder_remains_uninterpreted(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    path = tmp_path / "cylinder.stp"
    export_shape(BRepPrimAPI_MakeCylinder(5, 20).Shape(), path)
    result = extract_step(path)
    assert result["cylinder_candidates"][0]["diameter_mm"] == pytest.approx(10)
    assert result["cylinder_candidates"][0]["role"] == "unknown"


def test_declared_metre_units_convert_to_mm(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    path = tmp_path / "metre_box.stp"
    export_shape(BRepPrimAPI_MakeBox(1, 2, 3).Shape(), path)
    source = path.read_text()
    assert ".MILLI.,.METRE." in source
    path.write_text(source.replace(".MILLI.,.METRE.", "$,.METRE."))
    result = extract_step(path)
    assert result["measurements"]["aabb_large_mm"]["value"] == pytest.approx(3000, abs=1e-5)
    assert result["measurements"]["volume_mm3"]["value"] == pytest.approx(6e9)
    assert result["geometry_audit"]["file_length_units"] == ["metre"]
