import cv2
import numpy as np

def bbox_footprint(box, ratio=0.35):
    x1, y1, x2, y2 = map(float, box)
    y = y2 - max(0.0, y2-y1) * ratio
    return np.array([[x1,y],[x2,y],[x2,y2],[x1,y2]], dtype=np.float32)

def polygon_overlap(subject, polygon):
    if polygon is None or len(polygon) < 3:
        return 0.0
    subject = np.asarray(subject, np.float32)
    polygon = np.asarray(polygon, np.float32)
    area = abs(cv2.contourArea(subject))
    if area <= 1e-8:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(subject, polygon)
    return float(max(0.0, intersection) / area)

def grade_footprint(footprint, corridors, strips, high_threshold=0.01, medium_threshold=0.01):
    interior = max((polygon_overlap(footprint, p) for p in corridors), default=0.0)
    strip = max((polygon_overlap(footprint, p) for p in strips), default=0.0)
    if interior >= high_threshold:
        level = "high"
    elif strip >= medium_threshold:
        level = "medium"
    else:
        level = "low"
    return level, interior, strip

def point_in_polygons(point, polygons):
    return any(cv2.pointPolygonTest(np.asarray(p, np.float32), tuple(map(float, point)), False) >= 0 for p in polygons)
