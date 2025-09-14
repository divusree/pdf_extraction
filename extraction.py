import io, os, re
import gc
import json
import string
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple, Set

from pdfminer.layout import LAParams
from pdfminer.high_level import extract_pages, extract_text_to_fp
import pdfminer

import pandas as pd
from PIL import Image, ImageDraw
import fitz
import traceback
import chromadb
from sentence_transformers import SentenceTransformer


# origin at left top 
USABLE_REGION = [150, 50, 2730, 2120] #[140, 50, 2750, 2120]
SHEET_NUMBER = [2820, 2020, 2960, 2060]
SHEET_TITLE = [2760, 1885, 2960, 1980]
SHEET_DETAILS = [2750, 50, 2970, 2120]

def extract_everything_from_pdf(path, page_numbers, laparams, get_images = True):
    doc = None
    if isinstance(path, io.BytesIO):
        doc = fitz.open(stream = path)
    else:
        doc = fitz.open(path)
    images = {}
    layout = {}
    if doc:
        for page_id in page_numbers:
            for page_layout in extract_pages(path, laparams = laparams, page_numbers = [page_id] ):
                layout[page_id] = page_layout
                if get_images:
                    img = doc.load_page(page_id).get_pixmap(dpi=300).pil_image().resize((int(layout[page_id].width), int(layout[page_id].height)))
                    images[page_id] = img    
    print(f"pdfminer and image extraction done for {len(layout)} pages")
    return layout, images

def document_classifier(df):
    # ISSUE FOR PERMIT AND BID in df['text'] for y centroid  < 65 -> specs document.
    # else if there is are too many LTLines (find threshold) or sheet number present in document bottom right - thems the floor plans
    pass

def collect_section_info(layout):
    scope_misc_match = defaultdict(lambda : defaultdict(list))
    # toc_pages = []
    translator = str.maketrans("", "", string.punctuation.replace("_", "") + "–")
    csi_json = OrderedDict()
    footer_pattern = re.compile(r"(.+)\s*(\d{2} \d{2} \d{2})\s*-\s*(\d{1,2})")

    DIVISION_SEPARATOR = "GENERAL REQUIREMENTS"
    for page_no in layout.keys():
        df, _ = get_layout_dataframe(layout[page_no])
        has_footer = df[df['cy']> 710] # footer margins need more clarity 
        all_content = list(df[df['cy']>65].groupby(["cy"])["text"].apply(' '.join).reset_index()['text'].values) #switch to without footer df['cy'].between(65, 710)
        has_toc = df['text'].str.contains("table of contents", case = False).any()
        if len(has_footer)>0:
            # matches the pattern 'ACOUSTICAL VERTICAL FOLDING PANEL PARTITIONS   10 22 39 - 5'
            match = footer_pattern.match(' '.join(has_footer['text'].values))
            if match:
                subgroup = match.group(1).strip()
                scope_id = match.group(2)
                scope_misc_match[subgroup]["scope_id"] = scope_id
                scope_misc_match[subgroup]['pages'].append(page_no)                
        for content in all_content:
            if has_toc:
                c = content
                if "division" in c.lower():
                    match = re.match(r"\w*\s*(\d{2})", c)
                    if match:
                        div_num = c[match.span(0)[0]:match.span(0)[1]].split()[1].strip()
                        div_type = c[match.span(0)[1]:].translate(translator).strip()
                        csi_json[div_type] = {"div": div_num, "section": {}}
                else:
                    match = re.match(r"(\d{2} \d{2} \d{2})\s*\.*(.+)", c)
                    if match:
                        # print(match.groups(), match.group(2))
                        subgroup = match.group(2)
                        scope_id = match.group(1)
                        csi_json[div_type]["section"].update({subgroup: {"scope_id": scope_id, "pages": []}})

            else:
                # match anything that is like 
                # "SECTION 01 21 00 - ALLOWANCES"
                # "230548.01 - VIBRATION ISOLATION"
                # "230800 - SYSTEM COMMISSIONING"
                # "01785 - service and warranty (maintenance contract)"
                content_match = re.match(r"(SECTION\s)*(\d{2} \d{2} \d{2}|\d{6}|\d{5}|\d{6}\.d{2})\s*-\s*(.+)", content)
                if content_match:
                    # print(content_match.groups())
                    scope_misc_match[content_match.group(3)]["scope_id"] = content_match.group(2)
                    scope_misc_match[content_match.group(3)]['pages'].append(page_no)
                if DIVISION_SEPARATOR in content:
                    print("new division", page_no)  

    return scope_misc_match, csi_json

def create_collection(csi_json, collection_name):
    # model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    docs_to_embed = [f"{key} division contains {', '.join(list(value['scope'].keys()))}" if value.get("scope") else f"{key} division info not available"  for key, value in csi_json.items() ]

    chroma_client = chromadb.PersistentClient()
    collection = chroma_client.create_collection(name=collection_name)
    collection.upsert(
        ids = list(csi_json.keys()),
        documents= docs_to_embed
    )

    return collection    

def query_collection(query, collection):
    results = collection.query(
        query_texts=[query],  
        n_results=2  
    )    
    return results

def get_sections(df):
    matcher = re.compile(r'(?i)(?:\d{1,2}|\d{1,2}\.\d{1,2}|[a-z])\s*\.\s*(.+)')

    subsections = []
    subsection_bbox = []
    for i, row in df[df['cy'].between(65, 740)].iterrows():
        matches = matcher.match(row['text'])
        if matches:
            subsections.append(row['text'])
            subsection_bbox.append(row['bbox'])
        else:
            if len(subsections) == 0:
                subsection_bbox.append(row['bbox'])
                subsections.append(row['text'])
            else:
                subsections[-1] = subsections[-1] + ' ' + row['text']
                x0,y0,x1,y1 = subsection_bbox[-1]
                # this is weird - bboxes are weird from content.bboxes
                subsection_bbox[-1] = [min(x0, row['x0']),max(y0, row['y0']),max(x1, row['x1']),min(y1, row['y1'])]
    return subsections,subsection_bbox


# construction drawings functions
def get_centroid(bbox, base_round = 25):
    x0, y0, x1, y1 = bbox
    centroid_x = (x0 + x1) / 2
    centroid_y = (y0 + y1) / 2
    return base_round * round(centroid_x/base_round), base_round * round(centroid_y/base_round)
    
def lines_connected(line1, line2, threshold):
    x0_1, y0_1, x1_1, y1_1 = line1
    x0_2, y0_2, x1_2, y1_2 = line2
    
    endpoints1 = [(x0_1, y0_1), (x1_1, y1_1)]
    endpoints2 = [(x0_2, y0_2), (x1_2, y1_2)]
    
    for p1 in endpoints1:
        for p2 in endpoints2:
            distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
            if distance <= threshold:
                return True
    
    return False

def find_connected_component(lines, start_idx, visited, connection_threshold = 30):
    component = [start_idx]
    visited[start_idx] = True
    stack = [start_idx]
    
    while stack:
        current = stack.pop()
        for i, line in enumerate(lines):
            if not visited[i] and lines_connected(lines[current], line, connection_threshold):
                visited[i] = True
                component.append(i)
                stack.append(i)
    
    return component

def get_component_bbox(lines, component_indices):
    """Calculate bounding box for a component"""
    all_points = []
    for i in component_indices:
        x0, y0, x1, y1 = lines[i]
        all_points.extend([(x0, y0), (x1, y1)])
    
    if not all_points:
        return None
        
    min_x = min(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_x = max(p[0] for p in all_points)
    max_y = max(p[1] for p in all_points)
    
    return [min_x, min_y, max_x, max_y]

def is_bbox1_in_bbox2(bbox1, bbox2):
    x0, y0, x1, y1 = bbox1
    px0, py0, px1, py1 = bbox2
    if ((x0 > px0) and (y0 > py0)) and ((x1 < px1) and (y1 < py1)):
        return True
    return False

def get_layout_dataframe(page, base_round = 5):
    # origin at left top 
    all_bboxes = []
    for content in page:
        if isinstance(content, pdfminer.layout.LTTextBoxHorizontal):
            text = deepcopy(content.get_text()).replace('\n', " ").strip()
            if len(text) > 0:
                x0, y0, x1, y1 = content.bbox
                y0 = page.height - y0
                y1 = page.height - y1            
                cx, cy = get_centroid([x0, y0, x1, y1], base_round = base_round)
                is_sheet_details = not is_bbox1_in_bbox2([x0, y0, x1, y1], USABLE_REGION)
                all_bboxes.append({"text":text, "x0":x0, "y0": y0, "x1": x1,"y1":y1, "cx":cx, "cy": cy,
                                        "type": type(content), "bbox":[x0, y0, x1, y1], "is_sheet_details": is_sheet_details})
        elif isinstance(content, pdfminer.layout.LTLine):
            x0, y0, x1, y1 = content.bbox
            y0 = page.height - y0
            y1 = page.height - y1             
            cx, cy = get_centroid([x0, y0, x1, y1])
            is_sheet_details = not is_bbox1_in_bbox2([x0, y0, x1, y1], USABLE_REGION)
            if (get_length([x0, y0, x1, y1]) > 15):
                all_bboxes.append({"text":'', "x0":x0, "y0": y0, "x1": x1,"y1":y1, "cx":cx, "cy": cy,
                                   "type": type(content), "bbox":[x0, y0, x1, y1], "is_sheet_details": is_sheet_details})
            
        elif isinstance(content, pdfminer.layout.LTCurve) and not isinstance(content, (pdfminer.layout.LTRect, pdfminer.layout.LTLine)):
            x0, y0, x1, y1 = content.bbox
            y0 = page.height - y0
            y1 = page.height - y1             
            cx, cy = get_centroid([x0, y0, x1, y1])
            is_sheet_details = not is_bbox1_in_bbox2([x0, y0, x1, y1], USABLE_REGION)
            if (get_length([x0, y0, x1, y1]) > 15):
                all_bboxes.append({"text":'', "x0":x0, "y0": y0, "x1": x1,"y1":y1, "cx":cx, "cy": cy,
                                   "type": pdfminer.layout.LTLine, "bbox":[x0, y0, x1, y1], "is_sheet_details": is_sheet_details})

    df = pd.DataFrame(all_bboxes).sort_values(by = ["cy", "cx"], ascending = [True, True])
    return df, all_bboxes

def get_largest_table_bounds(lines, connection_threshold=30):
    if not lines:
        return None

    # Find all connected components
    visited = [False] * len(lines)
    components = []
    
    for i in range(len(lines)):
        if not visited[i]:
            component = find_connected_component(lines, i, visited, connection_threshold)
            components.append(component)
    
    largest_component = max(components, key=lambda comp: get_area(get_component_bbox(lines, comp)))

    return get_component_bbox(lines, largest_component)


def extract_sheet_locations(df):
    # get headers
    header_values = ['ABBREVIATIONS LEGEND', 'MATERIALS', 'SYMBOLS LEGEND', 'GENERAL PROJECT NOTES', 'PROJECT DATA', 'SHEET INDEX']
    # mean_hz_centroid = df[df['text'].isin(header_values)]['cy'].mean()
    headers = df[df['text'].isin(header_values)]
    roi_header = headers[headers['text'].str.contains("SHEET INDEX")]
    if len(roi_header) == 1:
        try:
            roi_header_dict = headers.loc[roi_header.index.item()]
            prev_header_col = headers.loc[roi_header.index.item()-1]
            possible_table_regions = df[(df["cx"].between(prev_header_col['x1'],roi_header_dict['x1']+200) ) & (df['type'] == pdfminer.layout.LTLine)]['bbox'].to_list()
            tx0, ty0, tx1, ty1 = get_largest_table_bounds(possible_table_regions)
            table_df =  df[(df["cx"].between(tx0, tx1) ) & (df["cy"].between(ty0, ty1) ) & (df['type'] == pdfminer.layout.LTTextBoxHorizontal)].sort_values(['cy', "cx"])
            sheet_values = list(table_df.groupby(["cy"])["text"].apply(list).reset_index()['text'].values)
            sheet_values = {val[0]:val[1] for val in sheet_values}

            return sheet_values
        except:
            traceback.print_exc()
    return None

def bbox_to_coords(bbox):
    left, top, right, bottom = [int(val) for val in bbox]
    return [(left, top), (right, top), (right, bottom), (left, bottom)]

def show_bboxes(image, display_bboxes, allow_display = True, origin = 'top'):
    image1copy = image.copy()
    imgdraw = ImageDraw.Draw(image1copy)  
    width, height = image.size
    for bbox in display_bboxes:
        x0,y0,x1, y1 = bbox
        if origin == "bottom":
            y0 = height - y0
            y1 = height - y1
        bbox = bbox_to_coords([x0,y0,x1, y1])
        imgdraw.polygon(bbox, outline ="red")
    if allow_display:
        display(image1copy)
        return
    return image1copy

def get_area(bbox):
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    return abs(width * height)

def get_length(bbox):
    x0, y0, x1, y1 = bbox
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    return max(width, height)

def bbox_overlap_area(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2    
    # Find intersection coordinates
    x_left = max(x1_1, x1_2)
    y_bottom = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_top = min(y2_1, y2_2)
    
    # Check if there's actual overlap
    if x_right <= x_left or y_top <= y_bottom:
        return 0.0
    
    # Calculate overlap area
    overlap_width = x_right - x_left
    overlap_height = y_top - y_bottom
    
    return overlap_width * overlap_height

def merge_overlapping_bboxes(bboxes, 
                            overlap_threshold: float = 0.1,
                            method: str = "union") -> List[Tuple[float, float, float, float]]:

    if not bboxes:
        return []
    
    # Find overlapping groups based on ratio threshold
    groups = []
    n = len(bboxes)
    
    # More sophisticated grouping based on overlap ratio
    visited = [False] * n
    
    for i in range(n):
        if visited[i]:
            continue
            
        # Start new group
        group = [i]
        visited[i] = True
        stack = [i]
        
        while stack:
            current = stack.pop()
            
            for j in range(n):
                if visited[j]:
                    continue
                
                # Check if bbox j overlaps significantly with any bbox in current group
                should_add = False
                for group_member in group:
                    overlap_ratio = bbox_overlap_ratio(bboxes[current], bboxes[j], "intersection")
                    if overlap_ratio > overlap_threshold:
                        should_add = True
                        break
                
                if should_add:
                    group.append(j)
                    visited[j] = True
                    stack.append(j)
        
        groups.append(group)
    
    # Merge each group
    merged_bboxes = []
    
    for group in groups:
        if len(group) == 1:
            merged_bboxes.append(bboxes[group[0]])
        else:
            if method == "union":
                merged_bbox = merge_bboxes_union([bboxes[i] for i in group])
            elif method == "weighted_average":
                merged_bbox = merge_bboxes_weighted([bboxes[i] for i in group])
            else:
                raise ValueError(f"Unknown merge method: {method}")
            
            merged_bboxes.append(merged_bbox)
    
    return merged_bboxes


def merge_bboxes_union(bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """
    Merge multiple bboxes by taking their union (minimum bounding rectangle)
    """
    if not bboxes:
        return (0, 0, 0, 0)
    
    x0_min = min(bbox[0] for bbox in bboxes)
    y0_min = min(bbox[1] for bbox in bboxes)
    x1_max = max(bbox[2] for bbox in bboxes)
    y1_max = max(bbox[3] for bbox in bboxes)
    
    return (x0_min, y0_min, x1_max, y1_max)


def merge_bboxes_weighted(bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """
    Merge multiple bboxes using area-weighted averaging
    """
    if not bboxes:
        return (0, 0, 0, 0)
    
    if len(bboxes) == 1:
        return bboxes[0]
    
    # Calculate weights based on area
    areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
    total_area = sum(areas)
    
    if total_area == 0:
        return merge_bboxes_union(bboxes)  # Fallback
    
    weights = [area / total_area for area in areas]
    
    # Weighted average of coordinates
    x0 = sum(bbox[0] * weight for bbox, weight in zip(bboxes, weights))
    y0 = sum(bbox[1] * weight for bbox, weight in zip(bboxes, weights))
    x1 = sum(bbox[2] * weight for bbox, weight in zip(bboxes, weights))
    y1 = sum(bbox[3] * weight for bbox, weight in zip(bboxes, weights))
    
    return (x0, y0, x1, y1)


# Helper functions
def bbox_overlap_area(bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
    """Calculate overlap area between two bboxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x_left = max(x1_1, x1_2)
    y_bottom = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_top = min(y2_1, y2_2)
    
    if x_right <= x_left or y_top <= y_bottom:
        return 0.0
    
    return (x_right - x_left) * (y_top - y_bottom)


def bbox_overlap_ratio(bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float],
                      method: str = "intersection") -> float:
    """Calculate overlap ratio between two bboxes"""
    overlap_area = bbox_overlap_area(bbox1, bbox2)
    
    if overlap_area == 0:
        return 0.0
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    if method == "intersection":
        return overlap_area / min(area1, area2) if min(area1, area2) > 0 else 0.0
    elif method == "union":
        union_area = area1 + area2 - overlap_area
        return overlap_area / union_area if union_area > 0 else 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

def get_diagrams(df, table_area_threshold, connection_threshold = 10, white_spaces = None):

    lines = df['bbox'].to_list()
    if white_spaces:
        lines.extend(white_spaces)
    visited = [False] * len(lines)
    components = []

    for i in range(len(lines)):
        if not visited[i]:
            component = find_connected_component(lines, i, visited, connection_threshold=connection_threshold)
            components.append(component)

    all_tables = defaultdict(dict)
    raw_bboxes = []
    for idx, component in enumerate(components):
        bbox = get_component_bbox(lines, component)
        raw_bboxes.append(bbox)
    
    merged_bboxes =  merge_overlapping_bboxes(raw_bboxes, overlap_threshold=0.18)

    for idx, bbox in enumerate(merged_bboxes):
        area =  get_area(bbox)
        if (table_area_threshold[0] < area) and (area < table_area_threshold[1]):
            all_tables[idx]['table_regions'] = bbox    
            all_tables[idx]['area'] = area
    return all_tables , raw_bboxes, merged_bboxes

def process_page(layout, page_no, images, is_sheet = True):
    sheet_values = None
    df, _ = get_layout_dataframe(layout[page_no], base_round=5)

    if (page_no == 2) and is_sheet:
        sheet_values = extract_sheet_locations(df)

    main_page_df = df
    extracted_sheet_number = ''
    extracted_sheet_title = ''
    table_bboxes = []
    if is_sheet:
        text_df = df[df["is_sheet_details"] == True]
        tx0, ty0, tx1, ty1 = SHEET_NUMBER
        extracted_sheet_number = text_df[(text_df['x0'] > tx0-5) & (text_df['x1'] < tx1+5) 
                            & (text_df['y0'] > ty0-5) & (text_df['y1'] < ty1+5)
                            ].sort_values(['cy','cx'], ascending = [True, True])['text'].values
        extracted_sheet_number = ' '.join(extracted_sheet_number)
        tx0, ty0, tx1, ty1 = SHEET_TITLE 
        extracted_sheet_title = text_df[(text_df['x0'] > tx0-5) & (text_df['x1'] < tx1+5) 
                            & (text_df['y0'] > ty0-5) & (text_df['y1'] < ty1+5)
                            ].sort_values(['cy','cx'], ascending = [True, True])['text'].values
        extracted_sheet_title = ' '.join(extracted_sheet_title)

        is_schedule = re.search(r"\b(notes|schedule|abbrev|symbols)\b",extracted_sheet_title.lower())
        main_page_df = df[df["is_sheet_details"] == False]

        if is_schedule:
            all_tables, raw_bboxes, merged_bboxes = get_diagrams(main_page_df, table_area_threshold = [7000.0, 5000000.0], connection_threshold = 30)
        else:
            all_tables, raw_bboxes, merged_bboxes = get_diagrams(main_page_df, table_area_threshold = [1000.0, 5000000.0], connection_threshold = 40)
        table_bboxes = [v['table_regions'] for k, v in all_tables.items()]

    else:
        subsections_content,subsections_bboxes = get_sections(df)
        has_footer = df[df['cy']> 710] # footer margins need more clarity 
        footer_pattern = re.compile(r"(.+)\s*(\d{2} \d{2} \d{2})\s*-\s*(\d{1,2})")
        if len(has_footer)>0:
            # matches the pattern 'ACOUSTICAL VERTICAL FOLDING PANEL PARTITIONS   10 22 39 - 5'
            match = footer_pattern.match(' '.join(has_footer['text'].values))
            if match:
                extracted_sheet_title = match.group(0)         
        table_bboxes = subsections_bboxes

    processed_image = show_bboxes(images[page_no], table_bboxes, allow_display = False)

    # print(processed_image.size)
    # print(table_bboxes)
    return {"image":processed_image, "extracted_sheet_number": extracted_sheet_number, 
            "extracted_sheet_title": extracted_sheet_title, "sheet_values": sheet_values,
            "table_regions":table_bboxes}

if __name__ == "__main__":
    path = "Input - Specifications.pdf"
    laparams=LAParams(all_texts=True, detect_vertical=False, 
                        line_overlap=0.5, char_margin=2000.0, 
                        line_margin=0.5, word_margin=2,
                        boxes_flow=1)    
    layout, images = extract_everything_from_pdf(path, page_numbers = None, laparams=laparams)
    scope_misc_match, csi_json = collect_section_info(layout)

    # extract sheet index from construction specs
    laparams=LAParams(all_texts=True, detect_vertical=False, 
                            line_overlap=0.1, char_margin=0.2, 
                            line_margin=0.1, word_margin=0.2,
                            boxes_flow=1)  
    path = "Input - Construction Drawings.pdf"
    page_numbers = [0]
    layout, images = extract_everything_from_pdf(path, page_numbers = page_numbers, laparams=laparams)    
    results = {}
    for num in page_numbers:
        results[num] = process_page(layout, num, images)
        