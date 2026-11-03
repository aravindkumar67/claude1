import numpy as np
from core.geometry import bbox_footprint, polygon_overlap, grade_footprint

def test_footprint_bottom_35_percent():
    p = bbox_footprint([10, 20, 30, 60], .35)
    assert p.tolist() == [[10.0,46.0],[30.0,46.0],[30.0,60.0],[10.0,60.0]]

def test_overlap_and_grade_high():
    fp = bbox_footprint([0,0,10,10],.35)
    polygon = np.array([[0,0],[10,0],[10,10],[0,10]],dtype=np.float32)
    assert polygon_overlap(fp, polygon) > .99
    level, interior, strip = grade_footprint(fp,[polygon],[],.01,.01)
    assert level == "high" and interior > .99 and strip == 0
