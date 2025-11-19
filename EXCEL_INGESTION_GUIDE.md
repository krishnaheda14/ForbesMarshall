# Excel Ingestion & Automatic Schema Mapping Feature

## Overview

This feature allows your CNC Scheduling System to automatically understand and process **arbitrary Excel files** from any industry. It uses a combination of heuristic rules and LLM (Gemini AI) reasoning to intelligently map Excel columns to a canonical scheduling schema.

## Architecture

```
Excel Upload → Sheet Parsing → Auto-Mapping → Human Confirmation → Transformation → Scheduling
```

### Components

1. **models.py** - Canonical scheduling schema (industry-agnostic)
2. **excel_ingestion.py** - Excel file loading and column analysis
3. **schema_mapping.py** - Automatic column mapping (heuristics + LLM)
4. **data_transformer.py** - Convert mapped data to CanonicalJob objects
5. **main.py** - New API endpoints for the complete workflow

## Canonical Schema

All uploaded data is converted to a universal `CanonicalJob` model:

### Required Fields:
- `job_id` (string) - Unique job identifier
- `processing_time` (float) - Processing time in hours

### Optional Fields:
- `operation_id` - For multi-stage jobs
- `machine` - Target machine/work center
- `due_date` - Job deadline
- `release_date` - Earliest start date
- `priority` - Priority class (A/B/C or HIGH/MEDIUM/LOW)
- `quantity` - Lot size
- `can_outsource` - Boolean flag
- `outsourcing_cost` - Cost to outsource
- `setup_time` - Changeover time
- `part_type` - Part number or SKU
- `material_type` - Material grade
- And more...

## API Endpoints

### 1. Upload Excel File
```
POST /api/excel/upload
Content-Type: multipart/form-data

Response:
{
  "status": "success",
  "filename": "jobs.xlsx",
  "sheet_names": ["Sheet1", "Production"],
  "message": "File loaded successfully"
}
```

### 2. Parse Sheet & Analyze Columns
```
POST /api/excel/parse
Content-Type: multipart/form-data

Parameters:
- file: Excel file
- sheet_name: Sheet to parse (optional, uses first sheet)
- sample_rows: Number of sample rows (default: 10)

Response:
{
  "status": "success",
  "columns": [
    {
      "column_name": "Job Number",
      "data_type": "object",
      "inferred_type": "job_id",
      "sample_values": ["J001", "J002", "J003"],
      "null_count": 0,
      "unique_count": 100
    },
    ...
  ],
  "sample_data": [{...}, {...}],
  "row_count": 100,
  "column_count": 8
}
```

### 3. Auto-Map Columns (AI-Powered)
```
POST /api/excel/auto-map
Content-Type: multipart/form-data

Parameters:
- file: Excel file
- sheet_name: Sheet name (optional)
- use_llm: Enable LLM mapping (default: true)

Response:
{
  "status": "success",
  "mappings": [
    {
      "excel_column": "Job Number",
      "canonical_field": "job_id",
      "confidence": 0.95,
      "source": "llm+heuristic",
      "reasoning": "Column contains unique job identifiers",
      "available_fields": ["job_id", "processing_time", ...]
    },
    ...
  ],
  "sheet_info": {
    "sheet_name": "Sheet1",
    "row_count": 100,
    "column_count": 8
  }
}
```

### 4. Transform Data with Confirmed Mappings
```
POST /api/excel/transform
Content-Type: multipart/form-data

Body:
{
  "mappings": {
    "Job Number": "job_id",
    "Runtime (hrs)": "processing_time",
    "Due Date": "due_date",
    "Machine": "machine",
    "Priority": "priority"
  },
  "save_as_template": false,
  "template_name": "customer_a_format"
}

Response:
{
  "status": "success",
  "jobs": [
    {
      "job_id": "J001",
      "processing_time": 2.5,
      "due_date": "2025-11-25T00:00:00",
      "machine": "M1",
      "priority": "HIGH",
      ...
    },
    ...
  ],
  "job_count": 98,
  "errors": ["Row 5: processing_time must be positive"],
  "warnings": ["Row 15: due_date is before release_date"],
  "message": "Transformed 98 jobs successfully"
}
```

## How It Works

### Layer A: Heuristic Mapping

Uses pattern matching on column names:

| Pattern | Maps To |
|---------|---------|
| "job", "order", "wo" | job_id |
| "proc", "runtime", "duration" | processing_time |
| "due", "deadline", "promise" | due_date |
| "release", "start", "avail" | release_date |
| "machine", "resource", "line" | machine |
| "priority", "class" | priority |
| "qty", "quantity", "lot" | quantity |

Plus data type checks:
- Date columns → likely due_date or release_date
- Numeric columns → likely processing_time or quantity
- Yes/No columns → likely can_outsource

### Layer B: LLM Mapping

Sends to Gemini AI:
```
Columns: ["Job Number", "Proc Time (hrs)", "Planned Finish", "Customer"]
Sample Data: [["J001", 2.5, "2025-11-25", "Acme Corp"], ...]

Task: Map each column to canonical fields

Available Fields:
- job_id: Unique job identifier
- processing_time: Processing time in hours
- due_date: Job due date
- customer: Customer name (metadata)
- ignore: Column to skip
```

LLM Response:
```json
{
  "column_mappings": {
    "Job Number": {
      "field": "job_id",
      "confidence": 0.95,
      "reasoning": "Unique identifier pattern"
    },
    "Proc Time (hrs)": {
      "field": "processing_time",
      "confidence": 0.9,
      "reasoning": "Numeric time value in hours"
    },
    "Customer": {
      "field": "ignore",
      "confidence": 0.8,
      "reasoning": "Metadata, not required for scheduling"
    }
  }
}
```

### Combined Mapping

Takes the higher confidence between heuristic and LLM:
- If LLM confidence > Heuristic → use LLM
- Mark source as "llm+heuristic" or "heuristic" or "llm"

## Frontend Integration (To Be Implemented)

### Suggested UI Flow:

1. **Upload Screen**
   - File picker for Excel
   - Sheet selector (if multiple sheets)
   - "Upload & Analyze" button

2. **Column Mapping Screen**
   ```
   | Excel Column    | Detected As      | Confidence | Source    | Correct Mapping ▼  |
   |-----------------|------------------|------------|-----------|---------------------|
   | Job Number      | job_id           | 95%        | LLM+Rules | [job_id ▼]          |
   | Proc Time       | processing_time  | 90%        | LLM       | [processing_time ▼] |
   | Customer Name   | ignore           | 80%        | LLM       | [ignore ▼]          |
   ```
   - Dropdown for each column to correct mapping
   - "Save as Template" checkbox
   - "Confirm & Transform" button

3. **Validation Screen**
   - Show errors and warnings
   - Preview transformed jobs (first 10)
   - "Fix Errors" or "Proceed to Schedule"

4. **Integration with Existing Scheduler**
   - Load transformed jobs into current system
   - Run heuristics (SPT, EDD, CR, etc.)
   - Show Gantt chart and metrics

## Feature Checklist

### Already Implemented in Your Project ✅
- [x] Dataset analysis and summary
- [x] Gantt chart visualization
- [x] Machine breakdown simulation
- [x] Adding new jobs manually
- [x] Outsourcing threshold configuration
- [x] AI insights (using Gemini)
- [x] Multiple scheduling heuristics (SPT, EDD, CR, PRIORITY)
- [x] Make-or-buy decisions
- [x] Metrics calculation (tardiness, utilization, etc.)

### New Features Added 🆕
- [x] Excel file upload and parsing
- [x] Automatic column type detection (heuristic)
- [x] LLM-powered schema understanding
- [x] Combined heuristic + LLM mapping
- [x] Human-in-the-loop mapping confirmation API
- [x] Data transformation with validation
- [x] Error and warning reporting
- [x] Canonical scheduling schema
- [x] Template saving capability (API ready)

### To Be Completed 🔨
- [ ] Frontend UI for upload workflow
- [ ] Column mapping confirmation UI
- [ ] Template management (save/load)
- [ ] Integration layer: CanonicalJob → CNC Scheduler format
- [ ] Multi-operation job support (operation sequences)
- [ ] Scenario comparison (side-by-side)
- [ ] What-if analysis UI

## Usage Example

### Sample Excel File Format

**Any of these formats will work:**

**Format 1 - Manufacturing:**
```
| Job ID | Part No | Runtime (hrs) | Due Date   | Machine | Priority |
|--------|---------|---------------|------------|---------|----------|
| J001   | P-123   | 2.5           | 2025-11-25 | M1      | HIGH     |
| J002   | P-456   | 1.8           | 2025-11-26 | M2      | MEDIUM   |
```

**Format 2 - Service Industry:**
```
| Order # | Customer   | Duration (min) | Deadline   | Resource | Class |
|---------|------------|----------------|------------|----------|-------|
| O-1001  | Acme Corp  | 150            | 2025-11-25 | Tech-A   | A     |
| O-1002  | Beta Inc   | 90             | 2025-11-26 | Tech-B   | B     |
```

**Format 3 - Custom:**
```
| WO Number | Time Needed | Target Completion | Work Center | Critical? |
|-----------|-------------|-------------------|-------------|-----------|
| WO-5001   | 3.2         | 11/25/2025        | Line 1      | Yes       |
| WO-5002   | 2.1         | 11/26/2025        | Line 2      | No        |
```

**All will be automatically understood and mapped!**

## Dependencies

Add to `backend/requirements.txt`:
```
openpyxl>=3.1.0  # For Excel file handling
```

## Configuration

Ensure your `.env` file has:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Testing

Test the auto-mapping with curl:

```bash
# 1. Upload and parse
curl -X POST http://localhost:8001/api/excel/parse \
  -F "file=@sample_jobs.xlsx" \
  -F "sheet_name=Sheet1"

# 2. Auto-map
curl -X POST http://localhost:8001/api/excel/auto-map \
  -F "file=@sample_jobs.xlsx" \
  -F "use_llm=true"

# 3. Transform with mappings
curl -X POST http://localhost:8001/api/excel/transform \
  -F "file=@sample_jobs.xlsx" \
  -F 'mappings={"Job ID":"job_id","Runtime (hrs)":"processing_time","Due Date":"due_date"}'
```

## Next Steps

1. **Install Dependencies**
   ```bash
   cd backend
   pip install openpyxl
   ```

2. **Restart Backend**
   - New modules will be loaded
   - New endpoints will be available

3. **Create Frontend UI**
   - Upload component
   - Mapping table component
   - Integration with existing dashboard

4. **Test with Real Data**
   - Try different Excel formats
   - Verify LLM mapping quality
   - Refine heuristic rules if needed

## Industry Adaptability

This system is designed to work with:
- **Manufacturing** - CNC jobs, production orders
- **Services** - Service tickets, appointments
- **Construction** - Project tasks, work orders
- **Healthcare** - Patient scheduling, procedures
- **Logistics** - Delivery routes, warehouse tasks
- **IT** - Support tickets, dev tasks

The canonical schema is flexible enough to handle any scheduling problem!
