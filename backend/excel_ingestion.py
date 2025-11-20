# backend/excel_ingestion.py
"""
Excel file ingestion and initial processing
Handles file upload, sheet selection, and basic data extraction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
import openpyxl
from fastapi import UploadFile, HTTPException


class ExcelIngestor:
    """Handles Excel file ingestion and preprocessing"""
    
    def __init__(self):
        self.df = None
        self.sheet_names = []
        self.column_info = {}
        self.file_content = None  # Store file content for later parsing
        self.filename = None  # Store filename to detect type
        
    async def load_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        Load Excel or CSV file and extract metadata
        
        Returns:
            - sheet_names: List of available sheets (for Excel) or ["Sheet1"] (for CSV)
            - file_info: Basic file information
        """
        try:
            # Read file content
            content = await file.read()
            filename = file.filename.lower()
            
            # Store for later parsing
            self.file_content = content
            self.filename = filename
            
            # Check if it's a CSV file
            if filename.endswith('.csv'):
                # For CSV, we treat it as a single sheet
                self.sheet_names = ["Sheet1"]
                return {
                    "status": "success",
                    "filename": file.filename,
                    "sheet_names": self.sheet_names,
                    "message": f"CSV file loaded successfully.",
                    "file_type": "csv"
                }
            else:
                # Handle Excel files
                # Get sheet names
                excel_file = openpyxl.load_workbook(BytesIO(content), read_only=True)
                self.sheet_names = excel_file.sheetnames
                excel_file.close()
                
                return {
                    "status": "success",
                    "filename": file.filename,
                    "sheet_names": self.sheet_names,
                    "message": f"File loaded successfully. Found {len(self.sheet_names)} sheet(s).",
                    "file_type": "excel"
                }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load file: {str(e)}")
    
    async def parse_sheet(
        self, 
        file: UploadFile, 
        sheet_name: Optional[str] = None,
        sample_rows: int = 10
    ) -> Dict[str, Any]:
        """
        Parse specific sheet (Excel) or CSV file and extract column information
        
        Args:
            file: Uploaded Excel or CSV file
            sheet_name: Name of sheet to parse (uses first sheet if None, ignored for CSV)
            sample_rows: Number of sample rows to return
            
        Returns:
            - columns: List of column names
            - data_types: Detected data types per column
            - sample_data: Sample rows
            - row_count: Total number of rows
        """
        try:
            # Use stored file content and filename
            if not self.file_content or not self.filename:
                raise HTTPException(status_code=400, detail="No file loaded. Please load a file first.")
            
            # Read CSV or Excel based on stored filename
            if self.filename.endswith('.csv'):
                self.df = pd.read_csv(BytesIO(self.file_content))
                actual_sheet_name = "Sheet1"
            else:
                # Read specific sheet for Excel
                if sheet_name:
                    self.df = pd.read_excel(BytesIO(self.file_content), sheet_name=sheet_name)
                    actual_sheet_name = sheet_name
                else:
                    self.df = pd.read_excel(BytesIO(self.file_content))
                    actual_sheet_name = "Sheet1"
            
            # Normalize column names
            self.df.columns = self.df.columns.str.strip()
            
            # Extract column information
            columns_info = []
            for col in self.df.columns:
                col_data = self.df[col]
                
                # Detect data type
                dtype = str(col_data.dtype)
                inferred_type = self._infer_semantic_type(col, col_data)
                
                # Get non-null sample values
                sample_values = col_data.dropna().head(5).tolist()
                
                # Calculate statistics
                null_count = col_data.isna().sum()
                unique_count = col_data.nunique()
                
                columns_info.append({
                    "column_name": col,
                    "data_type": dtype,
                    "inferred_type": inferred_type,
                    "sample_values": sample_values,
                    "null_count": int(null_count),
                    "unique_count": int(unique_count),
                    "is_numeric": pd.api.types.is_numeric_dtype(col_data),
                    "is_datetime": pd.api.types.is_datetime64_any_dtype(col_data)
                })
            
            # Get sample data rows
            sample_data = self.df.head(sample_rows).fillna("").to_dict(orient='records')
            
            return {
                "status": "success",
                "columns": columns_info,
                "sample_data": sample_data,
                "row_count": len(self.df),
                "column_count": len(self.df.columns),
                "sheet_name": actual_sheet_name
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
    
    def _infer_semantic_type(self, column_name: str, column_data: pd.Series) -> str:
        """
        Infer semantic meaning of column based on name and data
        
        Returns one of: job_id, processing_time, due_date, release_date, 
                       machine, priority, quantity, outsourcing, other
        """
        col_lower = column_name.lower()
        
        # Job ID patterns
        if any(term in col_lower for term in ['job', 'order', 'wo', 'work_order', 'job_id', 'order_id', 'number']):
            if 'number' in col_lower or 'id' in col_lower:
                return 'job_id'
        
        # Operation ID patterns
        if any(term in col_lower for term in ['operation', 'op_id', 'step', 'stage', 'sequence']):
            return 'operation_id'
        
        # Processing time patterns
        if any(term in col_lower for term in ['proc', 'process', 'runtime', 'duration', 'time', 'hours', 'minutes', 'hrs']):
            if pd.api.types.is_numeric_dtype(column_data):
                return 'processing_time'
        
        # Due date patterns (includes "planned finish", "finish date", etc.)
        if any(term in col_lower for term in ['due', 'deadline', 'promise', 'delivery', 'target', 'finish', 'completion', 'end']):
            if pd.api.types.is_datetime64_any_dtype(column_data) or 'date' in col_lower or 'finish' in col_lower:
                return 'due_date'
        
        # Release date patterns
        if any(term in col_lower for term in ['release', 'start', 'avail', 'ready', 'begin']):
            if pd.api.types.is_datetime64_any_dtype(column_data) or 'date' in col_lower:
                return 'release_date'
        
        # Machine patterns
        if any(term in col_lower for term in ['machine', 'resource', 'line', 'work_center', 'workcenter', 'equipment']):
            return 'machine'
        
        # Priority patterns
        if any(term in col_lower for term in ['priority', 'class', 'importance', 'urgency']):
            return 'priority'
        
        # Quantity patterns
        if any(term in col_lower for term in ['quantity', 'qty', 'lot', 'batch', 'amount']):
            if pd.api.types.is_numeric_dtype(column_data):
                return 'quantity'
        
        # Outsourcing cost patterns (specific check for cost)
        if any(term in col_lower for term in ['outsourc', 'subcontract', 'external']):
            if 'cost' in col_lower or 'price' in col_lower or 'rate' in col_lower:
                if pd.api.types.is_numeric_dtype(column_data):
                    return 'outsourcing_cost'
            else:
                return 'can_outsource'
        
        # Can outsource patterns (Yes/No type)
        if any(term in col_lower for term in ['can_outsource', 'outsourceable', 'allow_outsource']):
            return 'can_outsource'
        
        # Vendor patterns
        if any(term in col_lower for term in ['vendor', 'supplier', 'subcon']):
            return 'vendor_id'
        
        # Material/part patterns
        if any(term in col_lower for term in ['part', 'material', 'product', 'item']):
            return 'part_type'
        
        # Setup time patterns
        if any(term in col_lower for term in ['setup', 'changeover', 'tooling']):
            if pd.api.types.is_numeric_dtype(column_data):
                return 'setup_time'
        
        # Customer patterns
        if any(term in col_lower for term in ['customer', 'client', 'account']):
            return 'customer'
        
        return 'other'
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return the loaded DataFrame"""
        if self.df is None:
            raise ValueError("No data loaded. Call parse_sheet first.")
        return self.df.copy()


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to be code-friendly
    - Strip whitespace
    - Replace spaces with underscores
    - Remove special characters
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r'[^\w\s]', '_', regex=True)
        .str.replace(r'\s+', '_', regex=True)
        .str.replace(r'_+', '_', regex=True)
        .str.strip('_')
    )
    return df
