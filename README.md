# Challenge 1a: PDF Processing Solution (Offline Ready)

## Overview
This is a **sample solution** for Challenge 1a of the Adobe India Hackathon 2025. The challenge requires implementing a PDF processing solution that extracts structured data from PDF documents and outputs JSON files. The solution must be containerized using Docker and meet specific performance and resource constraints.

**🔒 This solution is configured to run completely offline** with predownloaded ML models.

## Official Challenge Guidelines

### Submission Requirements
- **GitHub Project**: Complete code repository with working solution
- **Dockerfile**: Must be present in the root directory and functional
- **README.md**:  Documentation explaining the solution, models, and libraries used

### Build Command
```bash
docker build --platform linux/amd64 -t <reponame.someidentifier> .
```

### Run Command
```bash
docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output/repoidentifier/:/app/output --network none <reponame.someidentifier>
```

### Critical Constraints
- **Execution Time**: ≤ 10 seconds for a 50-page PDF
- **Model Size**: ≤ 200MB (if using ML models)
- **Network**: No internet access allowed during runtime execution
- **Runtime**: Must run on CPU (amd64) with 8 CPUs and 16 GB RAM
- **Architecture**: Must work on AMD64, not ARM-specific

### Key Requirements
- **Automatic Processing**: Process all PDFs from `/app/input` directory
- **Output Format**: Generate `filename.json` for each `filename.pdf`
- **Input Directory**: Read-only access only
- **Open Source**: All libraries, models, and tools must be open source
- **Cross-Platform**: Test on both simple and complex PDFs

## Offline Implementation Features

### Pre-downloaded Models
- **SentenceTransformer Model**: `intfloat/e5-small` (~128MB) is downloaded during Docker build
- **No Runtime Downloads**: All dependencies are baked into the container
- **Network Isolation**: Runs with `--network none` for complete offline operation

### Model Details
- **Primary Model**: `intfloat/e5-small` for semantic text analysis
- **Size**: ~128MB (well within 200MB constraint) - documentation:https://arxiv.org/pdf/2212.03533
- **Purpose**: Document title selection and text analysis
- **Location**: Stored in `/app/model` within container

## Sample Solution Structure
```
Challenge_1a/
├── sample_dataset/
│   ├── outputs/         # JSON files provided as outputs.
│   ├── pdfs/            # Input PDF files
│   └── schema/          # Output schema definition
│       └── output_schema.json
├── Dockerfile           # Docker container configuration
├── process_pdfs.py      # Sample processing script
└── README.md           # This file
```

## Sample Implementation

### Current Sample Solution
The provided `process_pdfs.py` is a **basic sample** that demonstrates:
- PDF file scanning from input directory
- Dummy JSON data generation
- Output file creation in the specified format

**Note**: This is a placeholder implementation using dummy data. A real solution would need to:
- Implement actual PDF text extraction
- Parse document structure and hierarchy
- Generate meaningful JSON output based on content analysis

### Sample Processing Script (`process_pdfs.py`)
```python
# Current sample implementation
def process_pdfs():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Process all PDF files
    for pdf_file in input_dir.glob("*.pdf"):
        # Generate structured JSON output
        # (Current implementation uses dummy data)
        output_file = output_dir / f"{pdf_file.stem}.json"
        # Save JSON output
```

### Sample Docker Configuration (Offline Ready)
```dockerfile
FROM --platform=linux/amd64 python:3.10
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the processing script
COPY process_pdfs.py .

# Predownload the SentenceTransformer model to make it available offline
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('intfloat/e5-small'); model.save('/app/model')"

# Copy input and output directories
COPY input/ ./input/
COPY output/ ./output/

# Run the script
CMD ["python", "process_pdfs.py"]
```

## Building and Running

### Quick Start
```bash
# Build the image (downloads models during build)
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .

# Run completely offline
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
```

### Build Scripts
For convenience, use the provided build scripts:

**Linux/Mac:**
```bash
chmod +x build.sh
./build.sh
```

**Windows:**
```cmd
build.bat
```

## Expected Output Format

### Required JSON Structure
Each PDF should generate a corresponding JSON file that **must conform to the schema** defined in `sample_dataset/schema/output_schema.json`.

## 4. Solution Documentation

#### 1a. Outline Extraction Approach

The outline extraction process focuses on identifying structural headings within a PDF by analyzing various text properties and applying a multi-stage, heuristic-driven approach.

* **Header/Footer Detection:**
    * **Brief:** Headers and footers are not typically counted as headings. While they can contain text, their purpose is to provide information about the document as a whole, such as the page number, date, or document title, and they appear consistently on each page. Headings, on the other hand, are used to structure the content of the document and indicate the organization of the text, hence excluded from our heading extraction.
    * **Heuristic-Based Identification:** The solution first identifies recurring headers and footers by analyzing text segments in the top (0-15% of page height) and bottom (85-100% of page height) regions of the initial pages (up to 20 pages by default, `HEADER_FOOTER_SAMPLE_PAGES`).
    * **Spatial and Stylistic Fingerprinting:** Text in these regions is grouped into "spatial slots" based on rounded Y-coordinate, rounded font size, bold status, and rounded X-coordinate, with defined tolerances (`HEADER_FOOTER_Y_TOLERANCE_PX`, `HEADER_FONT_SIZE_TOLERANCE_PX`, `HEADER_X_COORD_TOLERANCE_PX`).
    * **Consistency Check:** A segment is considered a header/footer if it appears on a minimum number of pages (`HEADER_FOOTER_MIN_PAGES_REQUIRED`, default 2) and its occurrence ratio exceeds a threshold (`HEADER_FOOTER_THRESHOLD_PAGES_RATIO`, default 0.5) within the sampled pages.
    * **Dynamic Content Normalization:** A `normalize_hf_text_for_fingerprint` function replaces dynamic elements like page numbers, dates, times, versions, copyrights, and GUIDs with generic placeholders (e.g., `#PAGENUM#`, `#DATETIME#`) to create consistent fingerprints for robust matching.
    * **Fixed Substring Extraction:** For confirmed H/F slots, the `extract_fixed_substrings` function identifies common, recurring fixed substrings (minimum length 3, minimum occurrence ratio 0.7) to ensure accurate identification even with minor variations. If no strong fixed segments are found, aggressively normalized representative text is used as a fallback, provided it's not purely dynamic.
    * **Exclusion from Headings:** Any text identified as a header or footer is given an extremely low score (-10000) in the heading scoring phase, ensuring it's excluded from the final outline.

* **Heading Candidate Scoring:**
    * **Multi-Factor Scoring:** Each line of text is assigned a score based on several features:
        * **Font Size & Percentile:** Text with larger font sizes and higher percentile ranks (e.g., >=98th percentile) receive higher scores.
        * **Font Style:** Bold text is given a significant positive score (7 points).
        * **Text Characteristics:** The word count (1-10 words preferred, with an optimal range around 4 words), title-cased or all-uppercase text (for shorter lines), and absence of trailing punctuation (except for numbered headings) contribute to the score.
        * **Numbered Heading Patterns:** Specific regex patterns for Roman numerals, alphabetic, simple numeric, and multi-level numeric headings are used to identify and heavily score structured headings (e.g., 8 points for multi-level numeric, 7 for Roman numerals). Lines that look like simple list items are penalized.
        * **Vertical Spacing (Contextual):** Lines with significant vertical spacing above or below them (indicating separation from surrounding content) receive higher scores.
        * **Position on Page:** For the first page (page 0), content in the top half of the page receives a slight score boost (3 points).
    * **Noise Filtering:** General noise patterns (e.g., ellipses, lines with only punctuation, short uppercase section markers, citation numbers, figure/table captions) are penalized significantly (-5000) to filter them out from heading candidates.
    * **Adaptive Thresholding:** An adaptive threshold, combining the 70th percentile and mean plus 0.75 standard deviation of scores, is used to dynamically filter out less relevant candidates. Only candidates with scores above -2000 (after noise and H/F penalties) are considered.

* **Post-Processing for Outline Structure:**
    * **Multi-line Heading Merging:** Consecutive candidate lines on the same page are merged if they are vertically proximate (y-diff < 0.5 * avg line height), have similar font size (within 5% tolerance), bold status, and X-coordinate alignment (within 5px tolerance), and the subsequent line doesn't appear to start a new heading marker.
    * **Deduplication:** Merged candidates are deduplicated based on a normalized lowercase version of their text (removing common leading numbers/letters and excess whitespace), keeping the one with the highest score.   
    * **Hierarchical Level Inference:** Hierarchical levels (H1, H2, H3) are inferred based on distinct font size, X-coordinate alignment, and bold status. Styles with larger font sizes and less indentation are generally assigned higher levels. A significant font size drop (e.g., >8%) or X-coordinate indentation (e.g., >20px) can trigger a new, lower level. The system aims to identify up to 3 distinct levels.

* **Title Selection:**
Selecting the most appropriate document title is a critical step in the outline extraction process. The solution employs a multi-stage approach:

    * *Metadata Title Check:** If the PDF's metadata title exists and is semantically similar to the top heading candidates (and not considered noise), it is chosen as the document title.
    * **Semantic Centrality:** If the metadata title is missing or irrelevant, the system computes embeddings for all heading candidates and selects the most semantically central one (i.e., the heading most similar to all others) as the title.
    * **Fallback:** If no suitable candidates are found, the filename stem is used as the title.
    * **Exclusion from Outline:** The selected title is excluded from the final outline to avoid duplication.

    This ensures that the document title is both contextually relevant and distinct from the extracted headings.

    
* **Multilingual Approach:**
    * **Heuristic-Based Adaptability:** The fundamental heuristics employed, such as analyzing font size, bold status, vertical spacing, and X-coordinate alignment, are largely language-agnostic. These visual and layout cues are common indicators of headings across a wide array of languages and writing systems.
    * **Percentile-Based Analysis:** The percentile-based analysis of font sizes automatically adjusts to the spectrum of font sizes present within any given document, regardless of its textual language.
    * **Regular Expression Patterns:** While some `HEADING_PATTERNS` (e.g., for Roman numerals `^[IVXLCDM]+\.?\s+`, alphabetic `^[A-Z]\.?\s+`, simple numeric `^\d+\.?\s+`, and multi-level numeric `^\d+(\.\d+)+\.?\s+`) are designed with Western numbering and lettering conventions in mind, they are broadly applicable in many international documents.
    * **Language-Independent Noise Filtering:** The `SKIP_LINE_PATTERNS` used for filtering general noise (like ellipses or lines consisting only of punctuation) are generally independent of the specific language of the document.

### How the ML Model Is Utilized

The primary ML model used in Challenge 1a is the **SentenceTransformer** model, `intfloat/e5-small`. Its role is focused on enhancing the document's title selection through semantic analysis and filtering out poor quality headings:

- **Document Title Selection:** Within the `select_best_title` function, after all potential headings have been identified and deduplicated, the `intfloat/e5-small` model converts the text of these heading candidates and the document's metadata title into numerical vector embeddings.

- **Semantic Similarity:** These embeddings are compared using cosine similarity—both among the heading candidates and between each candidate and the metadata title. This process identifies the heading that is most semantically representative of the document's content, or determines if the metadata title is a suitable match for the main heading.

- **Multilingual Capability:** The selection of `intfloat/e5-small` as the embedding model enables multilingual support for semantic aspects of title selection. This model is explicitly designed to be multilingual, capable of generating semantically meaningful vector embeddings for texts in various languages. This facilitates accurate cosine similarity calculations when comparing different potential heading candidates or when comparing a chosen candidate title with the document's metadata title, even if they are expressed in different languages or use distinct linguistic phrasing for the same underlying concept.

### Libraries Used for 1a. Outline Extraction

#### Core PDF Processing
- **PyMuPDF (fitz)**: Primary PDF text extraction and font analysis. It provides access to text blocks, lines, and spans, including their bounding boxes, font sizes, and font names.
- **NumPy**: Used for numerical operations, such as calculating font size percentiles, means, and handling arrays of scores.

#### Text Processing and Pattern Recognition
- **re (Python's re module)**: Employed for regular expression matching to identify heading patterns (`HEADING_PATTERNS`), filter noise (`SKIP_LINE_PATTERNS`), and normalize text for header/footer fingerprinting. Patterns are designed to be broadly applicable across different languages.
- **sentence_transformers**: Primarily uses `intfloat/e5-small` model for semantic text analysis in title selection, with foundational multilingual capabilities supporting various languages through diverse training datasets.
- **torch**: The underlying deep learning framework used by `sentence_transformers` for model execution.

#### Language-Agnostic Processing Components
- **Font-based Analysis**: Visual cues (font size, bold status, spacing) that work across different writing systems and languages.
- **Layout Detection**: Spatial analysis (X-coordinate alignment, vertical spacing) that is independent of text language.
- **Pattern Recognition**: Heading patterns that are broadly applicable to international documents, particularly those following Western numbering conventions.

#### Standard Libraries
- **pathlib**: File system operations
- **json**: Output formatting
- **os**: Directory handling
- **collections** (`defaultdict`, `Counter`): Specialized data structures for grouping and counting
- **logging**: System monitoring and debugging
- 
### How to Build and Run Your Solution

#### Prerequisites
- Docker installed and running
- At least 8GB available RAM
- AMD64 architecture support

#### Build Process
```bash
# Navigate to the solution directory
cd Challenge-1a/

# Build the Docker image (downloads models during build)
docker build --platform linux/amd64 -t adobe-pdf-processor:hackathon2025 .

# Verify build completed successfully
docker images | grep adobe-pdf-processor
```

#### Running the Solution
```bash
# Prepare input directory with PDF files
mkdir -p input output

# Copy your PDF files to the input directory
cp your-pdfs/*.pdf input/

# Run the solution (completely offline)
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  --network none \
  adobe-pdf-processor:hackathon2025
```

#### Expected Execution Flow
1. Container starts and loads pre-downloaded models
2. Scans `/app/input` directory for PDF files
3. Processes each PDF using statistical analysis
4. Generates corresponding JSON files in `/app/output`
5. Logs processing statistics and completion status

#### Performance Verification
```bash
# Check processing time for large PDFs
time docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  --network none \
  adobe-pdf-processor:hackathon2025

# Verify output format
ls -la output/
cat output/sample-document.json | jq '.' # Pretty print JSON
```

#### Troubleshooting
- **Memory Issues**: Ensure Docker has access to at least 8GB RAM
- **Platform Issues**: Verify `--platform linux/amd64` is used in build command
- **Network Errors**: Confirm `--network none` flag is present during execution
- **Missing Output**: Check that input PDFs are readable and not corrupted

**Note**: This solution is designed to run completely offline with all dependencies and models pre-downloaded during the Docker build process.

## Implementation Guidelines

### Performance Considerations
- **Memory Management**: Efficient handling of large PDFs
- **Processing Speed**: Optimize for sub-10-second execution
- **Resource Usage**: Stay within 16GB RAM constraint
- **CPU Utilization**: Efficient use of 8 CPU cores

### Testing Strategy
- **Simple PDFs**: Test with basic PDF documents
- **Complex PDFs**: Test with multi-column layouts, images, tables
- **Large PDFs**: Verify 50-page processing within time limit


## Testing Your Solution

### Local Testing (Offline)
```bash
# Build the Docker image (this will download models during build)
docker build --platform linux/amd64 -t pdf-processor .

# Test with sample data (completely offline)
docker run --rm -v $(pwd)/sample_dataset/pdfs:/app/input:ro -v $(pwd)/test_output:/app/output --network none pdf-processor
```

### Validation Checklist
- [ ] All PDFs in input directory are processed
- [ ] JSON output files are generated for each PDF
- [ ] Output format matches required structure
- [ ] **Output conforms to schema** in `sample_dataset/schema/output_schema.json`
- [ ] Processing completes within 10 seconds for 50-page PDFs
- [ ] Solution works without internet access (`--network none`)
- [ ] Memory usage stays within 16GB limit
- [ ] Compatible with AMD64 architecture
- [ ] Models are predownloaded and < 200MB total

### Offline Verification
To verify the solution runs completely offline:
1. Build the image: `docker build --platform linux/amd64 -t test .`
2. Disconnect from internet or use `--network none`
3. Run: `docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none test`
4. Verify processing completes successfully

---

**Important**: This is a sample implementation. Participants should develop their own solutions that meet all the official challenge requirements and constraints. 
