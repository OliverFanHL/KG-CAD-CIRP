"""Small CAD-kernel backend; analytic cylinders are candidates, never inferred bores."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from .io import sha256, read_json, write_json


def extract_step(path):
    try:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
        try:
            from OCP.collections import Sequence_TCollection_AsciiString as UnitSequence
        except ImportError:  # OCP 7.x
            from OCP.TColStd import TColStd_SequenceOfAsciiString as UnitSequence
        from OCP.BRepBndLib import BRepBndLib
        from OCP.Bnd import Bnd_Box
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID
        from OCP.TopoDS import TopoDS
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
        from OCP.BRepCheck import BRepCheck_Analyzer
    except ImportError as exc:
        raise RuntimeError(f"STEP backend unavailable/incompatible ({exc}); install pip install -e '.[step]'") from exc
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError("STEP read failed")
    lengths, angles, solids = (UnitSequence() for _ in range(3))
    reader.FileUnits(lengths, angles, solids)
    units = [lengths.Value(i).ToCString() for i in range(1, lengths.Length() + 1)]
    reader.SetSystemLengthUnit(1.0)  # OCCT system length unit in millimetres, before transfer.
    if reader.TransferRoots() < 1:
        raise ValueError("STEP root transfer failed")
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError("Null shape")
    valid = bool(BRepCheck_Analyzer(shape).IsValid())
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    low, high = box.CornerMin(), box.CornerMax()
    x0, y0, z0 = low.X(), low.Y(), low.Z()
    x1, y1, z1 = high.X(), high.Y(), high.Z()
    dims = [x1-x0, y1-y0, z1-z0]
    surface = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, surface)
    volume = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume, True)
    counts = {}
    for name, kind in [("face_count", TopAbs_FACE), ("edge_count", TopAbs_EDGE), ("solid_count", TopAbs_SOLID)]:
        explorer, count = TopExp_Explorer(shape, kind), 0
        while explorer.More():
            count += 1
            explorer.Next()
        counts[name] = count
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    cylinders, planar, face_index = [], 0, 0
    while explorer.More():
        cast_face = getattr(TopoDS, "Face_s", None) or TopoDS.Face
        face = cast_face(explorer.Current())
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() == GeomAbs_Cylinder:
            cylinder = adaptor.Cylinder()
            cylinders.append({"candidate_id": f"face:{face_index}", "face_index": face_index,
                              "diameter_mm": 2 * cylinder.Radius(), "status": "computed",
                              "role": "unknown", "note": "analytic surface; not necessarily an internal/full/through bore"})
        elif adaptor.GetType() == GeomAbs_Plane:
            planar += 1
        face_index += 1
        explorer.Next()
    unit_known = bool(units)
    source = f"OCP STEPControl_Reader; {Path(path).name}; sha256:{sha256(path)}"
    def measured(value, unit):
        return {"value": float(value), "unit": unit if unit_known else "unknown",
                "status": "computed" if unit_known and valid else "computed_unverified", "source": source}
    measurements = {f"aabb_{axis}_mm": measured(d, "mm") for axis, d in zip("xyz", dims)}
    for name, value in zip(["aabb_small_mm", "aabb_mid_mm", "aabb_large_mm"], sorted(dims)):
        measurements[name] = measured(value, "mm")
    measurements["area_mm2"] = measured(surface.Mass(), "mm2")
    if valid and counts["solid_count"] == 1:
        measurements["volume_mm3"] = measured(volume.Mass(), "mm3")
    features = {**counts, "dim_small_large_ratio": min(dims)/max(dims),
                "dim_mid_large_ratio": sorted(dims)[1]/max(dims),
                "face_count_frac_plane": planar/max(face_index, 1),
                "face_count_frac_cylinder": len(cylinders)/max(face_index, 1)}
    if "volume_mm3" in measurements:
        features["bbox_fill_ratio"] = volume.Mass()/(dims[0]*dims[1]*dims[2])
    return {"measurements": measurements, "cylinder_candidates": cylinders,
            "geometry_audit": {"valid_shape": valid, "file_length_units": units, "system_length_unit": reader.SystemLengthUnit(),
                               "aabb_semantics": "file-coordinate envelope; arbitrary rotation not evaluated",
                               "volume_semantics": "only emitted for valid single-solid shape"}, "kernel_features": features}


def measure_catalog(catalog, limit=None):
    selected = catalog["parts"][:limit] if limit else catalog["parts"]
    errors = []
    for part in selected:
        try:
            data = extract_step(part["absolute_path"])
            part.update({k: v for k, v in data.items() if k != "kernel_features"})
            # Preserve a single consistent descriptor representation; kernel features are separate.
            part["kernel_features"] = data["kernel_features"]
            part["file_sha256"] = sha256(part["absolute_path"])
        except (ValueError, RuntimeError, TypeError) as exc:
            errors.append({"part_id": part["part_id"], "error": str(exc)})
    catalog["measurement_errors"] = errors
    return errors


def surface_mesh(path, deflection=0.03):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError("Reference STEP import failed")
    reader.SetSystemLengthUnit(1.0)
    if reader.TransferRoots() < 1:
        raise ValueError("Reference STEP transfer failed")
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.25, False)
    cast = getattr(TopoDS, "Face_s", None) or TopoDS.Face
    explorer, triangles, missing, surfaces = TopExp_Explorer(shape, TopAbs_FACE), [], 0, []
    while explorer.More():
        location = TopLoc_Location()
        face = cast(explorer.Current())
        adaptor = BRepAdaptor_Surface(face)
        info = {"face_index": len(surfaces), "surface_type": str(adaptor.GetType())}
        if adaptor.GetType() == GeomAbs_Cone:
            info.update(cone_semi_angle_rad=adaptor.Cone().SemiAngle(), reference_radius_mm=adaptor.Cone().RefRadius())
        if adaptor.GetType() == GeomAbs_Cylinder:
            info["radius_mm"] = adaptor.Cylinder().Radius()
        surfaces.append(info)
        mesh = BRep_Tool.Triangulation_s(face, location)
        if mesh is None:
            missing += 1
        else:
            transform = location.Transformation()
            nodes = []
            for i in range(1, mesh.NbNodes()+1):
                p = mesh.Node(i).Transformed(transform)
                nodes.append([p.X(), p.Y(), p.Z()])
            for i in range(1, mesh.NbTriangles()+1):
                triangles.append([nodes[j-1] for j in mesh.Triangle(i).Get()])
        explorer.Next()
    if not triangles:
        raise ValueError("Reference shape has no surface triangles")
    triangles = np.asarray(triangles)
    xyz = triangles.reshape(-1, 3)
    dims = np.ptp(xyz, axis=0)
    cast_vertex = getattr(TopoDS, "Vertex_s", None) or TopoDS.Vertex
    vertices, explorer = [], TopExp_Explorer(shape, TopAbs_VERTEX)
    while explorer.More():
        p = BRep_Tool.Pnt_s(cast_vertex(explorer.Current()))
        vertices.append([p.X(), p.Y(), p.Z()])
        explorer.Next()
    return triangles, {"surface_bounds_xyz_mm": dims.tolist(), "surface_large_mm": float(max(dims)),
                       "vertex_bounds_xyz_mm": np.ptp(vertices, axis=0).tolist() if vertices else None,
                       "triangles": len(triangles), "faces_missing_mesh": missing, "deflection_mm": deflection, "analytic_surfaces": surfaces,
                       "scope": "Fresh surface/vertex read from STEP in the same OCCT kernel; not an independent CAD system or manufacturing metrology."}


def draw_surface(ax, triangles, elev=25, azim=35):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    xyz = triangles.reshape(-1, 3)
    center = (xyz.max(0)+xyz.min(0))/2
    radius = max(float(np.ptp(xyz, axis=0).max()/2), 1e-9)*1.04
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals/np.maximum(lengths[:, None], 1e-15)
    shade = .48+.45*np.abs(normals @ np.asarray([.3, -.4, .866]))
    colors = np.clip(shade[:, None]*np.asarray([.34, .64, .78]), 0, 1)
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, linewidths=0, rasterized=True))
    for setter, middle in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        setter(middle-radius, middle+radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev, azim)
    ax.set_axis_off()




def render_pool_part(part, item, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    code = item["inspection_id"]
    triangles, evidence = surface_mesh(part["absolute_path"], .05)
    evidence.update(part_id=part["part_id"], source_sha256=sha256(part["absolute_path"]), inspection_id=code,
                    aabb_quarantined=any(k.startswith("aabb_") and m["status"] == "computed_unverified" for k,m in part["measurements"].items()))
    np.savez_compressed(output / f"{code}.npz", triangles=triangles)
    fig = plt.figure(figsize=(10, 3.5))
    for i,(elev,azim) in enumerate([(25,35),(15,135),(90,0)],1):
        draw_surface(fig.add_subplot(1,3,i,projection="3d"), triangles, elev, azim)
    fig.suptitle(f"{code} | largest rendered extent {evidence['surface_large_mm']:.2f} mm", fontsize=13)
    fig.tight_layout(rect=[0,0,1,.92])
    fig.savefig(output / f"{code}.png", dpi=150)
    plt.close(fig)
    write_json(output / f"{code}.json", evidence)


def _render_entry():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    request = read_json(args.request)
    render_pool_part(request["part"], request["item"], args.output)



if __name__ == "__main__":
    _render_entry()
