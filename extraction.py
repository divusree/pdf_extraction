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

# origin at left top 
USABLE_REGION = [140, 50, 2750, 2120]
SHEET_NUMBER = [2820, 2020, 2960, 2060]
SHEET_TITLE = [2760, 1885, 2960, 1980]
SHEET_DETAILS = [2750, 50, 2970, 2120]

def extract_everything_from_pdf(path, page_numbers, laparams ):
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
                img = doc.load_page(page_id).get_pixmap(dpi=300).pil_image().resize((int(layout[page_id].width), int(layout[page_id].height)))
                images[page_id] = img    
    print(f"pdfminer and image extraction done for {len(layout)} pages")
    return layout, images

def collect_section_info(layout):
    scope_misc_match = defaultdict(lambda : defaultdict(list))
    toc_pages = []
    translator = str.maketrans("", "", string.punctuation.replace("_", "") + "–")
    csi_json = {}

    DIVISION_SEPARATOR = "GENERAL REQUIREMENTS"
    for page_no in layout.keys():
        df, _ = get_layout_dataframe(layout[page_no])
        has_footer = df[df['cy']> 710]
        all_content = list(df[df['cy']>65].groupby(["cy"])["text"].apply(' '.join).reset_index()['text'].values)
        has_toc = df['text'].str.contains("table of contents", case = False).any()
        if has_toc:
            toc_pages.append(page_no)
        # else:
        # if is_toc
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


            content_match = re.match(r"(SECTION\s)*(\d{2} \d{2} \d{2}|\d{6}|\d{5})\s*-\s*(.+)", content)
            if content_match:
                # print(content_match.groups())
                scope_misc_match[content_match.group(3)]["scope_id"] = content_match.group(2)
                scope_misc_match[content_match.group(3)]['pages'].append(page_no)
            if DIVISION_SEPARATOR in content:
                print("new division", page_no)  

    return scope_misc_match, csi_json

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


def extract_sheet_locations(page):
    df, _ = get_layout_dataframe(page)
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

def show_bboxes(image, display_bboxes, display = True):
    image1copy = image.copy()
    imgdraw = ImageDraw.Draw(image1copy)  
    width, height = image.size
    for bbox in display_bboxes:
        x0,y0,x1, y1 = bbox
        bbox = bbox_to_coords([x0,y0,x1, y1])
        imgdraw.polygon(bbox, outline ="red")
    if display:
        # display(image1copy)
        return
    return image1copy

def get_area(bbox):
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    return width * height

def get_length(bbox):
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    return max(width, height)

def get_diagrams(df, table_area_threshold, connection_threshold = 10):

    lines = df['bbox'].to_list()
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
    
    merged_bboxes = []
    for bidx1, bbox1 in enumerate(raw_bboxes):
        for bidx2 in range(len(raw_bboxes)):
            bbox2 = raw_bboxes[bidx2] 
            if lines_connected(bbox1, bbox2, connection_threshold):
        
                # Update bounding box
                min_x = min(bbox1[0], bbox2[0])
                min_y = min(bbox1[1], bbox2[1])
                max_x = max(bbox1[2], bbox2[2])
                max_y = max(bbox1[3], bbox2[3])
                bbox1 = [min_x, min_y, max_x, max_y]
                raw_bboxes[bidx1] = bbox1 
                merged_bboxes.append(bbox1)
    for idx, bbox in enumerate(merged_bboxes):
        area =  get_area(bbox)
        if (table_area_threshold[0] < area) and (area < table_area_threshold[1]):
            all_tables[idx]['table_regions'] = bbox    
            all_tables[idx]['area'] = area
    return all_tables , raw_bboxes, merged_bboxes

def process_page(layout, page_no, images, is_sheet = True):
    sheet_values = None
    if (page_no == 2) and is_sheet:
        sheet_values = extract_sheet_locations(layout[page_no])
    df, _ = get_layout_dataframe(layout[page_no], base_round=5)
    main_page_df = df
    extracted_sheet_number = ''
    extracted_sheet_title = ''
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
    else:
        is_schedule = False # extract everything
    if is_schedule:
        all_tables, raw_bboxes, merged_bboxes = get_diagrams(main_page_df, table_area_threshold = [7000.0, 5000000.0], connection_threshold = 30)
    else:
        all_tables, raw_bboxes, merged_bboxes = get_diagrams(main_page_df, table_area_threshold = [1000.0, 5000000.0], connection_threshold = 40)
    table_bboxes = [v['table_regions'] for k, v in all_tables.items()]
    processed_image = show_bboxes(images[page_no], table_bboxes, display = False)
    # print(processed_image.size)
    # print(table_bboxes)
    return {"image":processed_image, "extracted_sheet_number": extracted_sheet_number, 
            "extracted_sheet_title": extracted_sheet_title, "sheet_values": sheet_values}

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