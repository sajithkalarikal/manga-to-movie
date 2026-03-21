import asyncio
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


async def write_bytes(path: Path, content: bytes) -> None:
    await asyncio.to_thread(path.write_bytes, content)


async def detect_panel_boxes(image_path: Path) -> list[tuple[int, int, int, int]]:
    return await asyncio.to_thread(_detect_panel_boxes_sync, image_path)


def _detect_panel_boxes_sync(image_path: Path) -> list[tuple[int, int, int, int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to open image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    height, width = gray.shape
    boxes = _split_by_gutters(gray, 0, 0, root_width=width)
    boxes = _refine_detected_boxes(gray, boxes, page_width=width, page_height=height)
    if len(boxes) < 2:
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (width * height) * 0.03
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < min_area:
                continue
            boxes.append((x, y, x + w, y + h))

    if len(boxes) < 2:
        if _should_treat_as_full_page(gray):
            boxes = [(0, 0, width, height)]
        else:
            boxes = _fallback_boxes(gray)

    boxes = _merge_spurious_top_row_splits(boxes, page_width=width, page_height=height)
    boxes = _repair_asymmetric_top_bottom_layout(gray, boxes, page_width=width, page_height=height)
    boxes = _merge_fragmented_lower_band(boxes, page_width=width, page_height=height)
    return [tuple(int(value) for value in box) for box in _sort_boxes(boxes, height)]


def _fallback_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = gray.shape
    panel_count = 4 if height >= width else 3
    segment_height = max(height // panel_count, 1)
    boxes = []
    for index in range(panel_count):
        top = index * segment_height
        bottom = height if index == panel_count - 1 else (index + 1) * segment_height
        boxes.append((0, top, width, bottom))
    return boxes


def _should_treat_as_full_page(gray: np.ndarray) -> bool:
    height, width = gray.shape
    page_area = height * width

    horizontal_split = _find_gutter_split(gray, axis=0)
    vertical_split = _find_gutter_split(gray, axis=1)
    if horizontal_split is not None or vertical_split is not None:
        return False

    thresholded = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 220, 255, cv2.THRESH_BINARY_INV)[1]
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(thresholded, connectivity=8)
    large_components = []
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < page_area * 0.02:
            continue
        large_components.append((x, y, w, h, int(area)))

    if len(large_components) <= 1:
        return True

    large_components.sort(key=lambda item: item[4], reverse=True)
    top_components = large_components[:3]
    combined_area = sum(component[4] for component in top_components) / max(page_area, 1)

    # A single splash/full-page panel usually behaves like one dominant connected
    # mass without any trustworthy internal gutter path. If the page is mostly
    # explained by one large component, prefer returning the whole page.
    largest_share = top_components[0][4] / max(page_area, 1)
    if largest_share >= 0.30 and combined_area <= 0.58:
        return True

    return False


def _sort_boxes(boxes: list[tuple[int, int, int, int]], height: int) -> list[tuple[int, int, int, int]]:
    row_band = max(height // 8, 1)
    return sorted(boxes, key=lambda box: (box[1] // row_band, -(box[0] + box[2]), box[1]))


def _sort_boxes_left_to_right(boxes: list[tuple[int, int, int, int]], height: int) -> list[tuple[int, int, int, int]]:
    row_band = max(height // 8, 1)
    return sorted(boxes, key=lambda box: (box[1] // row_band, box[0], box[1]))


def _merge_spurious_top_row_splits(
    boxes: list[tuple[int, int, int, int]],
    page_width: int,
    page_height: int,
) -> list[tuple[int, int, int, int]]:
    if len(boxes) < 4:
        return boxes

    remaining = _sort_boxes_left_to_right(boxes, page_height)
    merged = True
    while merged:
        merged = False
        for index in range(len(remaining) - 1):
            first = remaining[index]
            second = remaining[index + 1]
            if not _should_merge_top_row_pair(first, second, remaining, page_width=page_width, page_height=page_height):
                continue
            merged_box = (first[0], min(first[1], second[1]), second[2], max(first[3], second[3]))
            remaining = remaining[:index] + [merged_box] + remaining[index + 2 :]
            merged = True
            break
    return remaining


def _repair_asymmetric_top_bottom_layout(
    gray: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    page_width: int,
    page_height: int,
) -> list[tuple[int, int, int, int]]:
    if len(boxes) != 3:
        return boxes

    ordered = _sort_boxes_left_to_right(boxes, page_height)
    first, second, third = ordered
    if abs(first[1] - second[1]) > max(30, int(page_height * 0.02)):
        return boxes
    if abs(first[3] - second[3]) > max(30, int(page_height * 0.03)):
        return boxes
    if third[1] <= min(first[3], second[3]):
        return boxes

    top_width = second[2] - first[0]
    bottom_width = third[2] - third[0]
    if top_width < int(page_width * 0.90) or bottom_width < int(page_width * 0.94):
        return boxes

    anchor = (first[2] + second[0]) // 2
    vertical_split = _find_local_projection_valley(
        gray[third[1] : third[3], third[0] : third[2]],
        axis=1,
        anchor=anchor - third[0],
        span_limit=max(100, int(page_width * 0.14)),
    )
    if vertical_split is None:
        return boxes

    split_x = third[0] + vertical_split
    right_box = (split_x, third[1], third[2], third[3])
    left_box = (third[0], third[1], split_x, third[3])

    horizontal_split = _find_local_projection_valley(
        gray[left_box[1] : left_box[3], left_box[0] : left_box[2]],
        axis=0,
        anchor=int((left_box[3] - left_box[1]) * 0.55),
        span_limit=max(120, int(page_height * 0.12)),
    )

    repaired = [(first[0], min(first[1], second[1]), second[2], max(first[3], second[3]))]
    if horizontal_split is not None:
        split_y = left_box[1] + horizontal_split
        repaired.append((left_box[0], left_box[1], left_box[2], split_y))
        repaired.append((left_box[0], split_y, left_box[2], left_box[3]))
    else:
        repaired.append(left_box)
    repaired.append(right_box)
    return repaired


def _find_local_projection_valley(
    gray: np.ndarray,
    axis: int,
    anchor: int,
    span_limit: int,
) -> int | None:
    if gray.size == 0:
        return None

    if axis == 1:
        profile = np.mean(gray < 200, axis=0).astype(np.float32)
        span = gray.shape[1]
    else:
        profile = np.mean(gray < 200, axis=1).astype(np.float32)
        span = gray.shape[0]

    if span < 80:
        return None

    start = max(int(span * 0.15), anchor - span_limit)
    end = min(int(span * 0.85), anchor + span_limit)
    if end - start < 20:
        return None

    smooth = np.convolve(profile, np.ones(21, dtype=np.float32) / 21.0, mode="same")
    search = smooth[start:end]
    valley_index = int(start + np.argmin(search))
    valley_value = float(smooth[valley_index])

    left_band = smooth[max(start, valley_index - span_limit) : valley_index]
    right_band = smooth[valley_index + 1 : min(end, valley_index + span_limit)]
    if left_band.size == 0 or right_band.size == 0:
        return None

    left_peak = float(np.max(left_band))
    right_peak = float(np.max(right_band))
    if min(left_peak, right_peak) < 0.18:
        return None
    if valley_value / max(min(left_peak, right_peak), 1e-6) > 0.45:
        return None
    return valley_index


def _merge_fragmented_lower_band(
    boxes: list[tuple[int, int, int, int]],
    page_width: int,
    page_height: int,
) -> list[tuple[int, int, int, int]]:
    if len(boxes) < 6:
        return boxes

    sorted_boxes = _sort_boxes_left_to_right(boxes, page_height)
    upper_boxes = [box for box in sorted_boxes if box[1] < int(page_height * 0.60)]
    lower_boxes = [box for box in sorted_boxes if box[1] >= int(page_height * 0.60)]
    if len(upper_boxes) < 2 or len(lower_boxes) < 4:
        return boxes

    right_candidate = max(lower_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    right_width = right_candidate[2] - right_candidate[0]
    if right_width > int(page_width * 0.45) or right_candidate[0] < int(page_width * 0.55):
        return boxes

    left_fragments = [box for box in lower_boxes if box != right_candidate]
    if not left_fragments:
        return boxes

    fragment_right = max(box[2] for box in left_fragments)
    if fragment_right >= right_candidate[0]:
        return boxes

    fragment_top = min(box[1] for box in left_fragments)
    fragment_bottom = max(box[3] for box in left_fragments)
    if fragment_top > int(page_height * 0.72):
        return boxes
    if fragment_bottom < int(page_height * 0.92):
        return boxes

    primary_fragments = [box for box in left_fragments if (box[2] - box[0]) >= int(page_width * 0.12)]
    if not primary_fragments:
        primary_fragments = left_fragments

    narrow_edge_fragments = [
        box
        for box in primary_fragments
        if box[0] <= int(page_width * 0.02) and (box[2] - box[0]) <= int(page_width * 0.20)
    ]
    if len(narrow_edge_fragments) >= 2:
        filtered = [box for box in primary_fragments if box not in narrow_edge_fragments]
        if filtered:
            primary_fragments = filtered

    merged_left = (
        min(box[0] for box in primary_fragments),
        min(box[1] for box in primary_fragments),
        max(box[2] for box in primary_fragments),
        max(box[3] for box in primary_fragments),
    )
    return upper_boxes + [right_candidate, merged_left]




def _should_merge_top_row_pair(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    boxes: list[tuple[int, int, int, int]],
    page_width: int,
    page_height: int,
) -> bool:
    if abs(first[1] - second[1]) > max(30, int(page_height * 0.02)):
        return False
    if abs(first[3] - second[3]) > max(30, int(page_height * 0.03)):
        return False

    row_top = min(first[1], second[1])
    row_bottom = max(first[3], second[3])
    row_height = row_bottom - row_top
    if row_top > int(page_height * 0.08):
        return False
    if row_height < int(page_height * 0.28):
        return False

    gap = second[0] - first[2]
    if gap < 0 or gap > max(18, int(page_width * 0.02)):
        return False

    first_width = first[2] - first[0]
    second_width = second[2] - second[0]
    if min(first_width, second_width) > int(page_width * 0.36):
        return False
    if max(first_width, second_width) < int(page_width * 0.55):
        return False

    combined_width = second[2] - first[0]
    if combined_width < int(page_width * 0.92):
        return False

    lower_boxes = [box for box in boxes if box[1] >= row_bottom - 10]
    if len(lower_boxes) < 3:
        return False

    return True


def _refine_detected_boxes(
    gray: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    page_width: int,
    page_height: int,
) -> list[tuple[int, int, int, int]]:
    refined: list[tuple[int, int, int, int]] = []
    for box in boxes:
        refined.extend(_refine_box(gray, box, page_width=page_width, page_height=page_height, depth=0))
    return refined


def _refine_box(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    page_width: int,
    page_height: int,
    depth: int,
) -> list[tuple[int, int, int, int]]:
    if depth >= 2:
        return [box]

    x1, y1, x2, y2 = box
    region = gray[y1:y2, x1:x2]
    if region.size == 0:
        return [box]

    stage1_children = _split_by_gutters(region, 0, 0, preferred_axis=None, root_width=page_width)
    if len(stage1_children) > 1:
        absolute_children = [
            (x1 + child_x1, y1 + child_y1, x1 + child_x2, y1 + child_y2)
            for child_x1, child_y1, child_x2, child_y2 in stage1_children
        ]
        refined: list[tuple[int, int, int, int]] = []
        for child in absolute_children:
            refined.extend(_refine_box(gray, child, page_width=page_width, page_height=page_height, depth=depth + 1))
        return refined

    if _is_large_tall_column(box, page_width=page_width, page_height=page_height):
        refinement = _find_horizontal_column_split(region)
    else:
        if not _should_try_component_refinement(region, box, page_width=page_width, page_height=page_height):
            return [box]
        refinement = _find_directional_component_split(region, page_width=page_width)
        if refinement is None:
            refinement = _find_projection_component_split(region, page_width=page_width)
    if refinement is None:
        return [box]

    axis, start, end = refinement
    if axis == 1:
        children = [(x1, y1, x1 + start, y2), (x1 + end, y1, x2, y2)]
    else:
        children = [(x1, y1, x2, y1 + start), (x1, y1 + end, x2, y2)]

    refined_children: list[tuple[int, int, int, int]] = []
    for child in children:
        refined_children.extend(_refine_box(gray, child, page_width=page_width, page_height=page_height, depth=depth + 1))
    return refined_children


def _should_try_component_refinement(
    region: np.ndarray,
    box: tuple[int, int, int, int],
    page_width: int,
    page_height: int,
) -> bool:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    page_area = page_width * page_height
    if area < page_area * 0.10:
        return False
    if width >= page_width * 0.82:
        return False
    if width < height * 0.70:
        return False
    return True


def _is_large_tall_column(
    box: tuple[int, int, int, int],
    page_width: int,
    page_height: int,
) -> bool:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    page_area = page_width * page_height
    if area < page_area * 0.12:
        return False
    if width < page_width * 0.35:
        return False
    if height < page_height * 0.35:
        return False
    return height > width * 1.15


def _split_by_gutters(
    gray: np.ndarray,
    offset_x: int,
    offset_y: int,
    depth: int = 0,
    preferred_axis: int | None = None,
    root_width: int | None = None,
) -> list[tuple[int, int, int, int]]:
    height, width = gray.shape
    if root_width is None:
        root_width = width
    if depth >= 3 or width < 80 or height < 80:
        return [(offset_x, offset_y, offset_x + width, offset_y + height)]

    axes = [preferred_axis, 1 - preferred_axis] if preferred_axis is not None else [0, 1]
    for axis in axes:
        if axis == 1 and width < int(root_width * 0.60) and height < int(width * 0.95):
            continue
        split = _find_gutter_split(gray, axis=axis)
        if split is None:
            continue
        start, end = split
        next_axis = 1 - axis
        if axis == 1:
            left = gray[:, :start]
            right = gray[:, end:]
            if left.shape[1] >= 40 and right.shape[1] >= 40:
                return [
                    *_split_by_gutters(left, offset_x, offset_y, depth + 1, preferred_axis=next_axis, root_width=root_width),
                    *_split_by_gutters(right, offset_x + end, offset_y, depth + 1, preferred_axis=next_axis, root_width=root_width),
                ]
        else:
            top = gray[:start, :]
            bottom = gray[end:, :]
            if top.shape[0] >= 40 and bottom.shape[0] >= 40:
                return [
                    *_split_by_gutters(top, offset_x, offset_y, depth + 1, preferred_axis=next_axis, root_width=root_width),
                    *_split_by_gutters(bottom, offset_x, offset_y + end, depth + 1, preferred_axis=next_axis, root_width=root_width),
                ]

    return [(offset_x, offset_y, offset_x + width, offset_y + height)]


def _find_directional_component_split(gray: np.ndarray, page_width: int) -> tuple[int, int, int] | None:
    height, width = gray.shape
    if width >= int(page_width * 0.82):
        return None

    thresholded = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 220, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal = _find_directional_split_from_components(
        thresholded,
        kernel_size=(50, 1),
        axis=0,
        min_component_ratio=0.14,
    )
    vertical = _find_directional_split_from_components(
        thresholded,
        kernel_size=(1, 50),
        axis=1,
        min_component_ratio=0.08,
    )
    candidates = [candidate for candidate in [horizontal, vertical] if candidate is not None]
    if not candidates:
        return None
    axis, start, end, score = max(candidates, key=lambda item: item[3])
    return axis, start, end


def _find_horizontal_column_split(gray: np.ndarray) -> tuple[int, int, int] | None:
    thresholded = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 220, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal = _find_directional_split_from_components(
        thresholded,
        kernel_size=(50, 1),
        axis=0,
        min_component_ratio=0.08,
    )
    if horizontal is not None:
        axis, start, end, _ = horizontal
        return axis, start, end

    eroded = cv2.erode(thresholded, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)), iterations=1)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    height, width = gray.shape
    total_area = height * width
    components: list[tuple[int, int, int, int, int]] = []
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < total_area * 0.025:
            continue
        components.append((x, y, x + w, y + h, int(area)))

    horizontal_projection = _projection_split_for_axis(gray, components, axis=0)
    if horizontal_projection is None:
        return _find_profile_valley_split(gray, axis=0)
    axis, start, end, _ = horizontal_projection
    return axis, start, end


def _find_profile_valley_split(gray: np.ndarray, axis: int) -> tuple[int, int, int] | None:
    if axis == 0:
        span = gray.shape[0]
        profile = np.mean(gray < 200, axis=1).astype(np.float32)
        margin = max(80, int(span * 0.08))
        peak_window = max(140, int(span * 0.18))
    else:
        span = gray.shape[1]
        profile = np.mean(gray < 200, axis=0).astype(np.float32)
        margin = max(80, int(span * 0.08))
        peak_window = max(120, int(span * 0.16))

    if span <= margin * 2:
        return None

    smooth = np.convolve(profile, np.ones(31, dtype=np.float32) / 31.0, mode="same")
    search = smooth[margin : span - margin]
    if search.size == 0:
        return None

    valley_index = int(margin + np.argmin(search))
    valley_value = float(smooth[valley_index])
    left_peak = float(np.max(smooth[max(20, valley_index - peak_window) : valley_index]))
    right_peak = float(np.max(smooth[valley_index + 1 : min(span - 20, valley_index + peak_window)]))
    if min(left_peak, right_peak) < 0.14:
        return None
    valley_ratio = valley_value / max(min(left_peak, right_peak), 1e-6)
    if valley_ratio > 0.22:
        return None

    split_start = max(0, valley_index - 2)
    split_end = min(span, valley_index + 3)
    if not _split_balance_ok(split_start, split_end, span):
        return None
    return axis, split_start, split_end


def _find_directional_split_from_components(
    thresholded: np.ndarray,
    kernel_size: tuple[int, int],
    axis: int,
    min_component_ratio: float,
) -> tuple[int, int, int, float] | None:
    closed = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size),
        iterations=1,
    )
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    height, width = thresholded.shape
    total_area = height * width
    boxes: list[tuple[int, int, int, int, int]] = []
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < total_area * min_component_ratio:
            continue
        boxes.append((x, y, x + w, y + h, int(area)))

    if len(boxes) < 2:
        return None

    boxes.sort(key=lambda box: box[1 if axis == 0 else 0])
    largest_area = max(box[4] for box in boxes)
    best: tuple[int, int, int, float] | None = None
    span = height if axis == 0 else width
    for first, second in zip(boxes, boxes[1:]):
        if min(first[4], second[4]) < largest_area * 0.35:
            continue
        if axis == 0:
            start = first[3]
            end = second[1]
            overlap = min(first[2], second[2]) - max(first[0], second[0])
            min_overlap = int(width * 0.45)
        else:
            start = first[2]
            end = second[0]
            overlap = min(first[3], second[3]) - max(first[1], second[1])
            min_overlap = int(height * 0.45)
        if end <= start:
            continue
        if overlap < min_overlap:
            continue
        if not _split_balance_ok(start, end, span):
            continue
        score = (first[4] + second[4]) / max(end - start, 1)
        candidate = (axis, start, end, float(score))
        if best is None or candidate[3] > best[3]:
            best = candidate
    return best


def _find_projection_component_split(gray: np.ndarray, page_width: int) -> tuple[int, int, int] | None:
    height, width = gray.shape
    if width >= int(page_width * 0.82):
        return None

    thresholded = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 220, 255, cv2.THRESH_BINARY_INV)[1]
    eroded = cv2.erode(thresholded, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)), iterations=1)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    total_area = height * width

    components: list[tuple[int, int, int, int, int]] = []
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < total_area * 0.03:
            continue
        components.append((x, y, x + w, y + h, int(area)))

    vertical = _projection_split_for_axis(gray, components, axis=1)
    horizontal = _projection_split_for_axis(gray, components, axis=0)
    candidates = [candidate for candidate in [vertical, horizontal] if candidate is not None]
    if not candidates:
        return None
    axis, start, end, _ = max(candidates, key=lambda item: item[3])
    return axis, start, end


def _projection_split_for_axis(
    gray: np.ndarray,
    components: list[tuple[int, int, int, int, int]],
    axis: int,
) -> tuple[int, int, int, float] | None:
    if len(components) < 2:
        return None

    if axis == 1:
        span = gray.shape[1]
        comps = sorted(components, key=lambda item: item[0])
        centers = [((item[0] + item[2]) // 2) for item in comps]
        profile = np.mean(gray < 200, axis=0).astype(np.float32)
        min_gap = max(80, int(span * 0.12))
    else:
        span = gray.shape[0]
        comps = sorted(components, key=lambda item: item[1])
        centers = [((item[1] + item[3]) // 2) for item in comps]
        profile = np.mean(gray < 200, axis=1).astype(np.float32)
        min_gap = max(60, int(span * 0.10))

    smooth = np.convolve(profile, np.ones(31, dtype=np.float32) / 31.0, mode="same")
    best: tuple[int, int, int, float] | None = None
    largest_area = max(item[4] for item in comps)
    for left, right in zip(range(len(centers) - 1), range(1, len(centers))):
        left_area = comps[left][4]
        right_area = comps[right][4]
        if min(left_area, right_area) < largest_area * 0.45:
            continue
        start_center = centers[left]
        end_center = centers[right]
        if end_center - start_center < min_gap:
            continue
        valley_index = int(start_center + np.argmin(smooth[start_center:end_center]))
        valley_value = float(smooth[valley_index])
        left_peak = float(np.max(smooth[max(20, start_center - 80) : max(start_center + 1, valley_index)]))
        right_peak = float(np.max(smooth[min(valley_index + 1, end_center) : min(span - 20, end_center + 80)]))
        if min(left_peak, right_peak) <= 0:
            continue
        valley_ratio = valley_value / min(left_peak, right_peak)
        if valley_ratio > 0.88:
            continue
        split_start = max(0, valley_index - 2)
        split_end = min(span, valley_index + 3)
        if not _split_balance_ok(split_start, split_end, span):
            continue
        score = (left_peak + right_peak) / max(valley_value, 1e-6)
        candidate = (axis, split_start, split_end, score)
        if best is None or candidate[3] > best[3]:
            best = candidate
    return best


def _find_gutter_split(gray: np.ndarray, axis: int) -> tuple[int, int] | None:
    strict_split = _find_gutter_split_pass(gray, axis=axis, min_white_coverage=0.92, min_brightness=240, relaxed=False)
    if strict_split is not None:
        return strict_split
    if axis == 1 and gray.shape[1] < int(gray.shape[0] * 0.75):
        return None
    relaxed_thresholds = (0.70, 200) if axis == 1 else (0.60, 175)
    return _find_gutter_split_pass(
        gray,
        axis=axis,
        min_white_coverage=relaxed_thresholds[0],
        min_brightness=relaxed_thresholds[1],
        relaxed=True,
    )


def _find_gutter_split_pass(
    gray: np.ndarray,
    axis: int,
    min_white_coverage: float,
    min_brightness: float,
    relaxed: bool,
) -> tuple[int, int] | None:
    if axis == 1:
        coverage = np.mean(gray >= 245, axis=0)
        brightness = np.mean(gray, axis=0)
        length = gray.shape[0]
        span = gray.shape[1]
    else:
        coverage = np.mean(gray >= 245, axis=1)
        brightness = np.mean(gray, axis=1)
        length = gray.shape[1]
        span = gray.shape[0]

    gutter_mask = (coverage >= min_white_coverage) & (brightness >= min_brightness)
    margin = max(12, span // 30)
    if gutter_mask.size <= margin * 2:
        return None
    gutter_mask[:margin] = False
    gutter_mask[-margin:] = False

    runs = _mask_runs(gutter_mask)
    if not runs:
        return None

    if axis == 1:
        min_run = max(6, span // 240)
    elif relaxed:
        min_run = max(20, span // 90)
    else:
        min_run = max(10, span // 120)

    min_cross_coverage = min_white_coverage if relaxed else (0.75 if length >= 700 else 0.82)
    candidates: list[tuple[int, int, float]] = []
    for start, end in runs:
        width = end - start
        if width < min_run:
            continue
        max_relaxed_width = max(12, int(span * (0.06 if axis == 0 else 0.045)))
        if relaxed and width > max_relaxed_width:
            continue
        if not _split_balance_ok(start, end, span):
            continue
        coverage_score = float(np.mean(coverage[start:end]))
        if coverage_score < min_cross_coverage:
            continue
        edge_score = _edge_support_score(gray, start, end, axis=axis, relaxed=relaxed)
        if edge_score is None:
            continue
        if relaxed:
            score = (coverage_score * edge_score) / max(width, 1)
            min_relaxed_score = 0.01
            if score < min_relaxed_score:
                continue
        else:
            score = float(width)
        candidates.append((start, end, score))
    if not candidates:
        return None
    if axis == 0:
        best_start, best_end, _ = min(candidates, key=lambda item: item[0])
    else:
        best_start, best_end, _ = max(candidates, key=lambda item: item[2])
    return best_start, best_end


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _split_balance_ok(start: int, end: int, span: int) -> bool:
    left = start
    right = span - end
    min_side = max(40, int(span * 0.22))
    return left >= min_side and right >= min_side


def _edge_support_score(gray: np.ndarray, start: int, end: int, axis: int, relaxed: bool) -> float | None:
    band = 5 if relaxed else 3
    dark_threshold = 120
    if axis == 1:
        left_band = gray[:, max(0, start - band) : start]
        right_band = gray[:, end : min(gray.shape[1], end + band)]
    else:
        left_band = gray[max(0, start - band) : start, :]
        right_band = gray[end : min(gray.shape[0], end + band), :]

    if left_band.size == 0 or right_band.size == 0:
        return None

    left_dark = float(np.mean(left_band < dark_threshold))
    right_dark = float(np.mean(right_band < dark_threshold))
    min_dark = 0.12 if relaxed else 0.05
    edge_score = min(left_dark, right_dark)
    if edge_score < min_dark:
        return None
    return edge_score


async def crop_panels(upload_path: Path, boxes: list[tuple[int, int, int, int]], output_dir: Path) -> list[Path]:
    return await asyncio.to_thread(_crop_panels_sync, upload_path, boxes, output_dir)


def _crop_panels_sync(upload_path: Path, boxes: list[tuple[int, int, int, int]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_paths: list[Path] = []
    with Image.open(upload_path) as image:
        rgb_image = image.convert("RGB")
        for index, bbox in enumerate(boxes, start=1):
            cropped = rgb_image.crop(bbox)
            panel_path = output_dir / f"panel_{index:02d}.png"
            cropped.save(panel_path)
            panel_paths.append(panel_path)
    return panel_paths
