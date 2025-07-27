import os
import json
import re
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from collections import defaultdict, Counter

# --- Heading Extraction Heuristics ---
# General noise patterns, excluding page numbers which will be handled by H/F detection or more specific noise rules
SKIP_LINE_PATTERNS = [
    r'^\.{3,}$',            # Ellipses
    r'^\s*[\W_]+\s*$',      # Lines with only punctuation/symbols/underscores (e.g., "---")
    r'^[A-Z]{1,3}\s*$',     # Short uppercase lines (e.g., section markers 'I', 'II', 'A') - still useful for one-letter elements
    r'^\s*\[\s*\d+\s*\]\s*$', # Citation numbers like "[9]"
    r'^\s*(\([a-zA-Z0-9]+\)|\d+\))\s*$', # Generic list item markers like (a), (i), 1) - but if they are part of a heading number, HEADING_PATTERNS takes precedence
    r'^(Fig\.|Figure)\s*\d+[.:]?\s*', # Figure captions that might be mistaken for headings
    r'^Table\s*\d+[.:]?\s*',     # Table captions
]

BOLD_INDICATORS = ['bold', 'black', 'heavy', 'semibold', 'demi', 'demibold', 'extrabold']
ITALIC_INDICATORS = ['italic', 'oblique']

# More specific patterns for actual headings, distinguishing from list items
HEADING_PATTERNS = {
    'roman_numerals': r'^[IVXLCDM]+\.?\s+', # Roman numerals followed by space (e.g., I. Introduction)
    'alphabetic': r'^[A-Z]\.?\s+',        # Single uppercase letter followed by space (e.g., A. Section)
    'numeric_simple': r'^\d+\.?\s+',       # Single number followed by space (e.g., 1. Chapter)
    'numeric_multilevel': r'^\d+(\.\d+)+\.?\s+', # Multi-level numbers (e.g., 1.1, 2.1.3)
}

EMBED_MODEL = SentenceTransformer('./model')  # Load the pre-trained embedding model

# --- Header/Footer Detection Global Variables ---
# Store confirmed header/footer patterns
# Key: (normalized_text_segment_lower, rounded_y_avg, rounded_font_size_avg, is_bold, rounded_x_avg)
# Value: count of pages seen on (not used directly in matching, but for confidence)
HEADER_FOOTER_CANDIDATES = set() # Use a set for unique confirmed H/F segments

# Intermediate storage for H/F identification
# Key: (rounded_y, rounded_font_size, is_bold, rounded_x) -> Value: list of (page_num, raw_text)
HEADER_FOOTER_SPATIAL_SLOTS = defaultdict(lambda: defaultdict(list)) # slot_key -> page_num -> list of raw_text

# Configuration for H/F detection
HEADER_FOOTER_MIN_PAGES_REQUIRED = 2 # Minimum number of pages a pattern must appear on
HEADER_FOOTER_SAMPLE_PAGES = 20 # Max pages to scan for H/F identification (e.g., first 20 for better statistical robustness)

HEADER_FOOTER_Y_TOLERANCE_PX = 4 # Pixel tolerance for Y-coordinate comparison (slightly reduced for tighter clusters)
HEADER_FONT_SIZE_TOLERANCE_PX = 0.75 # Font size tolerance for H/F comparison
HEADER_X_COORD_TOLERANCE_PX = 6 # X-coordinate tolerance for H/F comparison
HEADER_FOOTER_THRESHOLD_PAGES_RATIO = 0.5

HEADER_REGION_START_Y_RATIO = 0.0 # From top of page
HEADER_REGION_END_Y_RATIO = 0.15 # End of header region (e.g., top 15%)
FOOTER_REGION_START_Y_RATIO = 0.85 # Start of footer region (e.g., bottom 15% - 100%)
FOOTER_REGION_END_Y_RATIO = 1.0 # To bottom of page

# Regex patterns for normalizing common dynamic parts within H/F for more robust matching
PAGE_INFO_REGEX_NORM = r'(page\s*\d+\s*(of\s*\d+)?|pg\.\s*\d+|-\s*\d+\s*-|\s*\d+\s*-\s*|^\s*\d+\s*$)' # Covers "page X", "page X of Y", "pg. X", "- X -", "X -" and standalone "X"
DATE_TIME_REGEX_NORM = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|(?:\d{1,2}:?\d{1,2}\s*(?:AM|PM)?)\b'
VERSION_NUM_REGEX_NORM = r'(v(?:ersion)?\s*\d+(?:\.\d+)*)' # e.g., "Version 1.0", "v2.3"
COPYRIGHT_REGEX_NORM = r'©|copyright|all rights reserved'
GUID_REGEX_NORM = r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b' # Matches GUIDs

def normalize_hf_text_for_fingerprint(text):
    """
    Normalizes text likely to appear in headers/footers by replacing variable parts with placeholders.
    This is for creating consistent fingerprints for structural identification.
    """
    normalized = text.strip().lower()

    # Replace dynamic elements with generic placeholders
    normalized = re.sub(PAGE_INFO_REGEX_NORM, '#PAGENUM#', normalized, flags=re.IGNORECASE)
    normalized = re.sub(DATE_TIME_REGEX_NORM, '#DATETIME#', normalized, flags=re.IGNORECASE)
    normalized = re.sub(VERSION_NUM_REGEX_NORM, '#VERSION#', normalized, flags=re.IGNORECASE)
    normalized = re.sub(COPYRIGHT_REGEX_NORM, '#COPYRIGHT#', normalized, flags=re.IGNORECASE)
    normalized = re.sub(GUID_REGEX_NORM, '#GUID#', normalized, flags=re.IGNORECASE)

    # Replace any remaining isolated numbers (e.g., serial numbers not part of page info)
    normalized = re.sub(r'\b\d+\b', '#NUM#', normalized)

    # Reduce multiple spaces to single space
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized

def extract_fixed_substrings(text_list, min_len=5, min_occurrence_ratio=0.7):
    """
    Analyzes a list of text strings (from a common H/F slot) to find recurring fixed substrings.
    """
    if not text_list:
        return []

    # Get common tokens / words first
    all_words = []
    for t in text_list:
        all_words.extend(re.findall(r'\b\w+\b', t.lower()))
    word_counts = Counter(all_words)
    common_words = {word for word, count in word_counts.items() if count / len(text_list) >= min_occurrence_ratio}

    fixed_segments = set()
    for i in range(len(text_list[0])):
        for j in range(i + min_len, len(text_list[0]) + 1):
            segment = text_list[0][i:j].lower()
            if not segment.strip() or len(segment.strip()) < min_len:
                continue

            # Check if this segment exists in a high percentage of other strings
            is_consistent = True
            count = 0
            for other_text in text_list:
                if segment in other_text.lower():
                    count += 1
            if count / len(text_list) >= min_occurrence_ratio:
                fixed_segments.add(segment.strip())

    # Also add combinations of common words
    if common_words:
        for text_line in text_list:
            current_segment = []
            for word in re.findall(r'\b\w+\b', text_line.lower()):
                if word in common_words:
                    current_segment.append(word)
                else:
                    if len(" ".join(current_segment)) >= min_len:
                        fixed_segments.add(" ".join(current_segment))
                    current_segment = []
            if len(" ".join(current_segment)) >= min_len:
                fixed_segments.add(" ".join(current_segment))

    return list(fixed_segments)

# --- Helper Functions (continued) ---

def compute_font_percentiles(spans):
    """
    Computes the percentile rank for each span's font size.
    Handles empty spans list gracefully.
    """
    if not spans:
        return {}
    sizes = np.array([s['size'] for s in spans])
    unique_sizes = np.sort(np.unique(sizes))
    size_to_percentile = {
        sz: int(100 * np.sum(sizes <= sz) / len(sizes)) for sz in unique_sizes
    }
    return {id(s): size_to_percentile.get(s['size'], 0) for s in spans}

def is_noise(text):
    """
    Checks if a text line is considered general document noise.
    These are patterns that are unlikely to ever be valid headings or consistent H/F.
    """
    t = text.strip()
    if len(t) < 2 and not t.isdigit(): return True # Filter very short strings unless they are single digits (e.g. part of 1.1)
    for pat in SKIP_LINE_PATTERNS:
        if re.match(pat, t): return True
    return False

def is_bold(font_name):
    """
    Checks if a font name indicates a bold style.
    """
    font_name_lower = font_name.lower().replace('-', '')
    return any(ind in font_name_lower for ind in BOLD_INDICATORS)

def get_adaptive_threshold(scores):
    """
    Calculates an adaptive threshold for heading scores.
    Uses a combination of percentile and mean/std deviation.
    """
    if not scores:
        return 0
    scores = np.array(scores)

    perc_threshold = np.percentile(scores, 70)
    mean_score = np.mean(scores)
    std_dev = np.std(scores)

    std_threshold = mean_score + 0.75 * std_dev

    return max(perc_threshold, std_threshold)

def score_line(text, span, pct_map, avg_size, prev_y, next_y, line_count, page_num, page):
    """
    Scores a line's candidacy as a heading, incorporating more features and nuances.
    """
    original_text = text # Keep original for case checks
    text = text.strip()

    # --- Header/Footer Exclusion First and Foremost ---
    line_y = span['bbox'][1]
    line_font_size = span['size']
    line_is_bold = is_bold(span['font'])
    line_x = span['bbox'][0]

    for hf_key in HEADER_FOOTER_CANDIDATES:
        hf_norm_segment, hf_y_approx, hf_font_size_approx, hf_is_bold_status, hf_x_approx = hf_key

        # Match on approximate Y, font size, bold status, and X-coordinate
        # AND if the normalized segment is found within the current line's text
        # This is the crucial part for flexible H/F matching
        if abs(hf_y_approx - line_y) <= HEADER_FOOTER_Y_TOLERANCE_PX and \
           abs(hf_font_size_approx - line_font_size) <= HEADER_FONT_SIZE_TOLERANCE_PX and \
           abs(hf_x_approx - line_x) <= HEADER_X_COORD_TOLERANCE_PX and \
           hf_is_bold_status == line_is_bold and \
           hf_norm_segment in normalize_hf_text_for_fingerprint(original_text): # Use normalize_hf_text_for_fingerprint for content check
            return -10000 # Extremely low score to ensure removal from candidates

    # If it's not a confirmed H/F, check if it's general noise
    if is_noise(text):
        return -5000 # Very low score for general noise, less than H/F

    score = 0
    size = span['size']
    pct = pct_map.get(id(span), 0)
    font_name = span['font']

    # --- Font Size and Percentile ---
    if pct >= 98: score += 12
    elif pct >= 90: score += 9
    elif pct >= 80: score += 6
    elif pct >= 65: score += 3
    else: score -= 2

    ratio = size / avg_size if avg_size else 1
    score += (ratio - 1) * 8

    # --- Font Style ---
    if is_bold(font_name):
        score += 7

    # --- Text Characteristics ---
    wc = len(text.split())
    if 1 <= wc <= 10:
        score += (5 - abs(wc - 4))
    else:
        score -= abs(wc - 7) * 0.5

    # Case (use original_text for case checks)
    if original_text.istitle() and wc > 1 and not original_text.isupper():
        score += 4
    elif original_text.isupper() and wc <= 8:
        score += 5
    elif original_text.isupper() and wc > 8:
        score -= 3

    # Trailing punctuation
    if text.endswith('.') and not re.match(r'^\d+(\.\d+)*\.$', text):
        score -= 5
    if text.endswith((',', ';', ':')): score -= 3

    # Presence of numbering/markers
    is_numbered_heading = False
    for pat_type, pat in HEADING_PATTERNS.items():
        if re.match(pat, text):
            is_numbered_heading = True
            if pat_type == 'numeric_multilevel': score += 8
            elif pat_type == 'roman_numerals': score += 7
            elif pat_type == 'numeric_simple': score += 6
            elif pat_type == 'alphabetic': score += 5
            break

    # Penalize if it looks like a simple list item marker rather than a heading
    if not is_numbered_heading:
        if re.match(r'^\s*(\d+|\([a-z]\)|\([A-Z]\)|\([ivx]+\)|\.)\s+.*', text.strip()):
            score -= 5 # e.g., "1. This is a list item."
        if re.match(r'^\s*(\d+\.){1,}\s*$', text.strip()):
            score -= 10 # e.g., "1.1." alone on a line

    # --- Vertical Spacing (Contextual) ---
    if prev_y is not None and (span['bbox'][1] - prev_y) > avg_size * 2.5:
        score += 4
    elif prev_y is not None and (span['bbox'][1] - prev_y) > avg_size * 1.8:
        score += 2

    if next_y is not None and (next_y - (span['bbox'][1] + span['bbox'][3] - span['bbox'][1])) > avg_size * 1.8:
        score += 3

    # Position on page (for the *first* page only, as later pages might have lower headings if title spans multiple)
    if page_num == 0 and span['bbox'][1] < page.rect.height / 2:
        score += 3

    return score

def process_page(page, num, page_height, page_width):
    td = page.get_text('dict')

    spans = []
    for b in td['blocks']:
        for l in b.get('lines', []):
            spans.extend(l['spans'])

    pct_map = compute_font_percentiles(spans)

    lines = []
    line_groups = defaultdict(list)
    for b in td['blocks']:
        for l in b.get('lines', []):
            if l['spans']:
                line_y = round(np.mean([s['bbox'][1] for s in l['spans']]), 2)
                line_groups[line_y].extend(l['spans'])

    for y, line_spans in sorted(line_groups.items()):
        valid_spans = [s for s in line_spans if s['text'].strip()]
        if not valid_spans:
            continue
        valid_spans.sort(key=lambda s: s['bbox'][0])
        text = ''.join(s['text'] for s in valid_spans).strip()
        primary_span = max(valid_spans, key=lambda s: s['size']) if valid_spans else None

        if lines and text == lines[-1][1]:
            continue
        if primary_span:
            lines.append((y, text, primary_span))

    lines.sort(key=lambda x: x[0])

    if spans:
        all_sizes = np.array([s['size'] for s in spans])
        lower_bound = np.percentile(all_sizes, 25)
        upper_bound = np.percentile(all_sizes, 75)

        body_text_sizes = [s['size'] for s in spans if lower_bound <= s['size'] <= upper_bound]
        avg_size = np.mean(body_text_sizes) if body_text_sizes else np.mean(all_sizes)
    else:
        avg_size = 0

    cands, scores = [], []
    for i, (y, text, span) in enumerate(lines):
        prev_y = lines[i-1][0] if i > 0 else None
        next_y = lines[i+1][0] if i < len(lines)-1 else None

        sc = score_line(text, span, pct_map, avg_size, prev_y, next_y, len(lines), num, page) # num is 0-based page index

        # Only add candidates with a score higher than the general noise penalty (-5000)
        # This implicitly filters out H/F (-10000) and general noise.
        if sc > -2000: # Threshold above noise and H/F penalties
            cands.append({'text': text, 'page': num, 'score': sc, 'font_size': span['size'], 'y_coord': y, 'span': span, 'x_coord': span['bbox'][0]})
            scores.append(sc)

    thr = get_adaptive_threshold(scores) if scores else 0

    filtered_cands = []
    for c in cands:
        # Filter based on adaptive threshold. Scores below -2000 are already excluded.
        if c['score'] >= thr:
            filtered_cands.append(c)

    return filtered_cands

def merge_multi_line_headings(candidates, threshold_y_diff_ratio=0.5, font_size_tolerance_ratio=0.05, x_coord_tolerance=5):
    """
    Attempts to merge consecutive candidate lines that appear to be a single multi-line heading.
    Merges based on:
    - Same page
    - Vertical proximity (y_coord difference is small, relative to font size)
    - Similar font size and bold status
    - Similar X-coordinate (alignment)
    - The subsequent line typically starts with a lowercase letter or doesn't look like a new heading marker.
    """
    if not candidates:
        return []

    candidates.sort(key=lambda x: (x['page'], x['y_coord'], x['x_coord']))

    merged_candidates = []
    i = 0
    while i < len(candidates):
        current_cand = candidates[i].copy()

        if 'span' not in current_cand:
            merged_candidates.append(current_cand)
            i += 1
            continue

        current_span = current_cand['span']
        is_current_bold = is_bold(current_span['font'])

        j = i + 1
        while j < len(candidates):
            next_cand = candidates[j]
            if 'span' not in next_cand: break

            next_span = next_cand['span']

            if current_cand['page'] != next_cand['page']: break

            y_diff = next_cand['y_coord'] - (current_cand['y_coord'] + current_span['bbox'][3] - current_span['bbox'][1])
            avg_line_height = (current_span['size'] + next_span['size']) / 2

            font_size_similar = abs(current_cand['font_size'] - next_cand['font_size']) / current_cand['font_size'] < font_size_tolerance_ratio
            bold_status_similar = is_current_bold == is_bold(next_span['font'])
            x_coord_similar = abs(current_cand['x_coord'] - next_cand['x_coord']) < x_coord_tolerance

            next_text_starts_lower_or_no_new_heading_marker = (
                next_cand['text'][0].islower() or
                (not any(re.match(p, next_cand['text']) for p in HEADING_PATTERNS.values()))
            )

            if (y_diff < avg_line_height * threshold_y_diff_ratio and
                font_size_similar and
                bold_status_similar and
                x_coord_similar and
                next_text_starts_lower_or_no_new_heading_marker
            ):
                current_cand['text'] += " " + next_cand['text']
                current_span = next_span
                current_cand['span'] = current_span
                j += 1
            else:
                break

        merged_candidates.append(current_cand)
        i = j

    for cand in merged_candidates:
        if 'span' in cand:
            del cand['span']
        # Do NOT delete 'x_coord' here

    return merged_candidates


def select_best_title(candidates, meta_title):
    """
    Selects the best document title from candidates, prioritizing metadata if relevant,
    and using semantic similarity for overall coherence.
    """
    if not candidates and not meta_title:
        return "Untitled Document"

    texts = [c['text'] for c in candidates]

    if not texts and meta_title and len(meta_title) > 3 and not is_noise(meta_title):
        return meta_title

    if not texts:
        return "Untitled Document"

    emb = EMBED_MODEL.encode(texts, convert_to_tensor=True)
    sims = util.cos_sim(emb, emb).sum(dim=1)
    best_idx = int(torch.argmax(sims))
    best_heading_candidate_text = texts[best_idx]

    if meta_title and len(meta_title) > 3 and not is_noise(meta_title):
        try:
            mt_emb = EMBED_MODEL.encode(meta_title, convert_to_tensor=True)
            h_emb = EMBED_MODEL.encode(best_heading_candidate_text, convert_to_tensor=True)

            if util.cos_sim(mt_emb, h_emb).item() > 0.75:
                return meta_title
        except Exception as e:
            pass

    return best_heading_candidate_text

def infer_heading_levels(outline_candidates):
    """
    Infers hierarchical levels (H1, H2, H3) for headings based on font size,
    x-coordinate, and explicit numbering patterns.
    """
    if not outline_candidates:
        return []

    outline_candidates.sort(key=lambda x: (x['page'], x['y_coord'], x['x_coord']))

    unique_styles = defaultdict(lambda: {'texts': [], 'x_coords': [], 'font_sizes': []})
    for h in outline_candidates:
        key = (round(h['font_size'], 1), round(h['x_coord'] / 10) * 10, is_bold(h.get('span', {}).get('font', '')))
        unique_styles[key]['texts'].append(h['text'])
        unique_styles[key]['x_coords'].append(h['x_coord'])
        unique_styles[key]['font_sizes'].append(h['font_size'])

    sorted_styles = sorted(unique_styles.keys(), key=lambda k: (-k[0], k[1]))

    level_map = {}
    assigned_levels = []

    for i, style_key in enumerate(sorted_styles):
        font_size, x_coord_group, bold_status = style_key

        if not assigned_levels:
            level_map[style_key] = 'H1'
            assigned_levels.append('H1')
        else:
            last_assigned_style_key = next(k for k, v in level_map.items() if v == assigned_levels[-1])
            last_font_size, last_x_coord_group, _ = last_assigned_style_key

            font_size_drop = (last_font_size - font_size) / last_font_size if last_font_size else 0
            x_coord_indentation = (x_coord_group - last_x_coord_group)

            is_new_level = False
            if font_size_drop > 0.08 and len(assigned_levels) < 3: # Significant font size drop (e.g., >8%)
                is_new_level = True
            elif x_coord_indentation > 20 and len(assigned_levels) < 3: # Significant indentation (e.g., >20px)
                is_new_level = True

            if is_new_level:
                if assigned_levels[-1] == 'H1':
                    level_map[style_key] = 'H2'
                    assigned_levels.append('H2')
                elif assigned_levels[-1] == 'H2':
                    level_map[style_key] = 'H3'
                    assigned_levels.append('H3')
                else: # Fallback if more than 3 distinct levels identified by font/indentation
                    level_map[style_key] = 'H3'
            else: # If not a new level, assign to the lowest current level
                level_map[style_key] = assigned_levels[-1]

    final_outline = []
    for h in outline_candidates:
        best_match_key = None
        min_diff = float('inf')

        current_style_font_size = round(h['font_size'], 1)
        current_style_x_coord = round(h['x_coord'] / 10) * 10
        current_style_bold_status = is_bold(h.get('span', {}).get('font', ''))

        for style_key in sorted_styles:
            style_font_size, style_x_coord, style_bold_status = style_key

            diff = abs(current_style_font_size - style_font_size) * 10 + \
                   abs(current_style_x_coord - style_x_coord) * 0.5 + \
                   (0 if current_style_bold_status == style_bold_status else 5)

            if diff < min_diff:
                min_diff = diff
                best_match_key = style_key

        level = 'H3' # Default fallback
        if best_match_key and best_match_key in level_map:
            level = level_map[best_match_key]
        elif assigned_levels:
            level = assigned_levels[-1]

        final_outline.append({'level': level, 'text': h['text'], 'page': h['page']})

    return final_outline

# --- Header/Footer Identification Function ---
def identify_headers_footers(doc):
    """
    Analyzes the first few pages of a document to identify recurring headers and footers.
    Populates the global HEADER_FOOTER_CANDIDATES with detailed fingerprints.
    """
    global HEADER_FOOTER_CANDIDATES # Declare intent to modify global variable

    HEADER_FOOTER_CANDIDATES.clear() # Reset for each document

    page_sample_size = min(doc.page_count, HEADER_FOOTER_SAMPLE_PAGES)

    # Dictionary to store slot_key -> {page_num: [list_of_raw_texts_in_slot_on_this_page]}
    # slot_key: (rounded_y, rounded_font_size, is_bold, rounded_x)
    spatial_slot_texts = defaultdict(lambda: defaultdict(list))

    for i in range(page_sample_size):
        page = doc.load_page(i)
        page_rect = page.rect
        page_height = page_rect.height

        header_top_y_boundary = page_height * HEADER_REGION_START_Y_RATIO
        header_bottom_y_boundary = page_height * HEADER_REGION_END_Y_RATIO
        footer_top_y_boundary = page_height * FOOTER_REGION_START_Y_RATIO
        footer_bottom_y_boundary = page_height * FOOTER_REGION_END_Y_RATIO

        blocks = page.get_text('dict')['blocks']

        for b in blocks:
            for l in b.get('lines', []):
                for s in l.get('spans', []):
                    raw_text = s['text'].strip()
                    if not raw_text:
                        continue

                    span_y = s['bbox'][1]
                    span_font_size = s['size']
                    span_is_bold = is_bold(s['font'])
                    span_x = s['bbox'][0]

                    is_in_header_region = header_top_y_boundary <= span_y <= header_bottom_y_boundary
                    is_in_footer_region = footer_top_y_boundary <= span_y <= footer_bottom_y_boundary

                    if is_in_header_region or is_in_footer_region:
                        # Create a spatial-stylistic fingerprint for the slot
                        slot_fingerprint = (
                            round(span_y, 0), # Round Y to nearest integer
                            round(span_font_size, 0), # Round font size to nearest integer
                            span_is_bold,
                            round(span_x, 0) # Round X to nearest integer
                        )
                        spatial_slot_texts[slot_fingerprint][i].append(raw_text)

    # Now, analyze the collected texts within each spatial slot to find consistent H/F elements
    for slot_fingerprint, pages_data in spatial_slot_texts.items():
        pages_with_content = len(pages_data)

        # If content in this slot appears on enough pages
        if pages_with_content >= HEADER_FOOTER_MIN_PAGES_REQUIRED and \
           pages_with_content / page_sample_size >= HEADER_FOOTER_THRESHOLD_PAGES_RATIO:

            all_texts_in_slot = []
            for page_num in pages_data:
                all_texts_in_slot.extend(pages_data[page_num])

            # Use the first text line found in this consistent slot as a reference for normalization
            # This is a heuristic: assuming the text structure is broadly similar
            if all_texts_in_slot:
                representative_text = all_texts_in_slot[0]

                # Extract fixed common substrings from all texts in this slot
                # These fixed parts, combined with the slot fingerprint, form the H/F candidate
                # Filter out numbers and dates from fixed segments to prevent false positives if they aren't fully normalized
                fixed_segments = extract_fixed_substrings(all_texts_in_slot, min_len=3, min_occurrence_ratio=0.7) # Min length of 3 for segments

                # If no strong fixed segments, try with aggressive normalization of the representative text
                if not fixed_segments:
                    # Fallback: if no strong common fixed substrings, then use the normalized representative text
                    # but only if it's not purely dynamic (e.g., just page numbers after normalization)
                    normalized_rep_text = normalize_hf_text_for_fingerprint(representative_text)
                    if normalized_rep_text not in ['#pagenum#', '#datetime#', '#version#', '#num#', '#copyright#', '#guid#', '']:
                        fixed_segments.append(normalized_rep_text)

                for segment in fixed_segments:
                    # Add each robustly identified segment with its spatial-stylistic info as a H/F candidate
                    # The segment itself is what we'll check for inclusion in future lines
                    HEADER_FOOTER_CANDIDATES.add((
                        segment,
                        slot_fingerprint[0], # rounded_y
                        slot_fingerprint[1], # rounded_font_size
                        slot_fingerprint[2], # is_bold
                        slot_fingerprint[3]  # rounded_x
                    ))


# Main outline extraction
def extract_outline_from_pdf(path):
    doc = fitz.open(path)
    meta = doc.metadata.get('title', '')

    # Step 0: Identify headers/footers first
    identify_headers_footers(doc)

    all_cands = []
    # Iterate using 0-based page index
    for i in range(doc.page_count):
        pg = doc.load_page(i)
        page_height = pg.rect.height
        page_width = pg.rect.width
        all_cands.extend(process_page(pg, i, page_height, page_width)) # Pass 0-based index 'i'

    doc.close()

    # Step 1: Handle multi-line headings
    all_cands_merged = merge_multi_line_headings(all_cands)

    # Step 2: De-duplicate merged candidates, keeping the one with the highest score
    seen_texts = {}
    for c in sorted(all_cands_merged, key=lambda x: -x['score']):
        t_lower = c['text'].lower().strip()
        if not t_lower: continue
        # Normalized text for deduplication: remove common leading numbers/letters and excess whitespace
        # This is for deduplication of candidate headings, not H/F.
        normalized_text = re.sub(r'^\s*(\d+(\.\d+)*\s*|[A-Z]\.?\s*|[IVXLCDM]+\.?\s*)\s*', '', t_lower).strip()

        if normalized_text not in seen_texts:
            seen_texts[normalized_text] = c
        else:
            if c['score'] > seen_texts[normalized_text]['score']:
                seen_texts[normalized_text] = c

    uniq = list(seen_texts.values())
    uniq.sort(key=lambda x: (x['page']-1, x['y_coord'], x['x_coord']))

    title = select_best_title(uniq, meta) if uniq or meta else Path(path).stem

    outline_candidates = [h for h in uniq if h['text'].strip() != title.strip()]

    # Step 3: Infer multi-level headings
    final_outline = infer_heading_levels(outline_candidates)

    return {'title': title, 'outline': final_outline}

if __name__ == '__main__':
    os.makedirs('./input/', exist_ok=True)
    os.makedirs('./output/', exist_ok=True)

    print("Processing PDFs in 'input' directory...")
    processed_count = 0

    for f in os.listdir('input'):
        if f.lower().endswith('.pdf'):
            pdf_path = os.path.join('input', f)
            print(f"   Processing {f}...")
            try:
                res = extract_outline_from_pdf(pdf_path)
                output_filename = Path(f).stem + '.json'
                with open(os.path.join('output', output_filename), 'w', encoding='utf-8') as fw:
                    json.dump(res, fw, indent=4, ensure_ascii=False)
                print(f"    Successfully extracted outline to {output_filename}")
                processed_count += 1
            except Exception as e:
                print(f"    Error processing {f}: {e}")
                import traceback
                traceback.print_exc()

    if processed_count == 0:
        print("\nNo PDFs found in the 'input' directory or an error occurred during processing.")
        print("Please place PDF files in the 'input' folder and run the script again.")
    else:
        print(f"\nFinished processing {processed_count} PDF(s).")