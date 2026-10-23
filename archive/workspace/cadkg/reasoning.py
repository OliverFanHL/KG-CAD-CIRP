"""Three-valued evidence checks. No category-to-compatibility inference."""
from __future__ import annotations


def measurement_for(part, constraint):
    attr = constraint["attribute"]
    if attr in {"bore_diameter_mm", "shaft_diameter_mm"}:
        role = "bore" if attr.startswith("bore") else "shaft"
        candidates = [i for i in part["interfaces"] if i["role"] == role
                      and (not constraint.get("interface_id") or i["interface_id"] == constraint["interface_id"])]
        if len(candidates) != 1:
            return None, "missing or ambiguous reviewed interface"
        interface = candidates[0]
        return {"value": interface["diameter_mm"], "unit": "mm", "status": "curated", "source": interface["source"]}, None
    measurement = part["measurements"].get(attr)
    if not measurement:
        return None, "measurement missing"
    if measurement["status"] not in {"computed", "curated"} or measurement["unit"] != constraint["unit"]:
        return None, "measurement/unit not verified"
    return measurement, None


def check_constraints(part, constraints):
    import math
    def leq(a, b):
        # Absorb floating arithmetic noise, not a manufacturing or fit tolerance.
        return a <= b or math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
    checks = []
    for constraint in constraints:
        measurement, reason = measurement_for(part, constraint)
        if measurement is None:
            checks.append({"constraint": constraint, "status": "unknown", "reason": reason})
            continue
        value, op, bound = measurement["value"], constraint["op"], constraint["value"]
        passed = leq(value, bound) if op == "<=" else leq(bound, value) if op == ">=" else leq(bound[0], value) and leq(value, bound[1])
        checks.append({"constraint": constraint, "status": "pass" if passed else "fail", "measurement": measurement})
    states = {c["status"] for c in checks}
    status = "fail" if "fail" in states else "unknown" if "unknown" in states else "pass" if checks else "not_requested"
    return {"status": status, "checks": checks, "claim": "only the explicitly checked geometric requirements"}


def clearance_screen(catalog, bore_part_id, bore_interface_id, shaft_part_id, shaft_interface_id, minimum, maximum):
    import math
    if not all(math.isfinite(x) for x in [minimum, maximum]) or minimum > maximum:
        raise ValueError("Invalid allowed clearance interval")
    by_id = {p["part_id"]: p for p in catalog["parts"]}
    if bore_part_id not in by_id or shaft_part_id not in by_id:
        raise ValueError("Unknown part")
    def select(pid, iid, role):
        return next((i for i in by_id[pid]["interfaces"] if i["interface_id"] == iid and i["role"] == role), None)
    bore = select(bore_part_id, bore_interface_id, "bore")
    shaft = select(shaft_part_id, shaft_interface_id, "shaft")
    result = {"status": "unknown", "scope": "diameter-clearance screen only", "allowed_clearance_mm": [minimum, maximum],
              "assembly_compatibility": "unknown", "unresolved": ["axial engagement", "alignment", "loads", "materials", "surface finish", "other interfaces"]}
    if bore is None or shaft is None:
        result["reason"] = "Reviewed bore/shaft interface is missing"
        return result
    nominal = bore["diameter_mm"] - shaft["diameter_mm"]
    result["nominal_clearance_mm"] = nominal
    result["evidence"] = [bore["source"], shaft["source"]]
    if any(i.get("min_mm") is None or i.get("max_mm") is None for i in [bore, shaft]):
        result["reason"] = "Nominal dimensions only; tolerance bounds missing"
        return result
    lower, upper = bore["min_mm"] - shaft["max_mm"], bore["max_mm"] - shaft["min_mm"]
    result["clearance_interval_mm"] = [lower, upper]
    if lower >= minimum and upper <= maximum:
        result["status"] = "pass"
    elif upper < minimum or lower > maximum:
        result["status"] = "fail"
    else:
        result["reason"] = "Tolerance interval only partly satisfies requested range"
    return result
