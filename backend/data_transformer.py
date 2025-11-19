# backend/data_transformer.py
"""
Transform mapped Excel data to canonical Job objects
Handles type conversion, validation, and error reporting
"""

import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from models import CanonicalJob


class DataTransformer:
    """Transforms raw Excel data to canonical Job objects"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def transform(
        self, 
        df: pd.DataFrame, 
        column_mappings: Dict[str, str]
    ) -> Tuple[List[CanonicalJob], List[str], List[str]]:
        """
        Transform DataFrame to list of CanonicalJob objects
        
        Args:
            df: Source DataFrame
            column_mappings: Dict mapping df column names to canonical field names
            
        Returns:
            (jobs, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        jobs = []
        
        # Create reverse mapping (canonical -> excel column)
        reverse_map = {v: k for k, v in column_mappings.items() if v != 'ignore'}
        
        # Check for required fields
        if 'job_id' not in reverse_map:
            self.errors.append("Missing required field: job_id")
            return [], self.errors, self.warnings
        
        if 'processing_time' not in reverse_map:
            self.errors.append("Missing required field: processing_time")
            return [], self.errors, self.warnings
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                job_data = self._extract_job_data(row, reverse_map)
                
                # Validate and create job
                job = CanonicalJob(**job_data)
                jobs.append(job)
                
            except Exception as e:
                error_msg = f"Row {idx + 2}: {str(e)}"  # +2 for Excel row number (header + 0-index)
                self.errors.append(error_msg)
        
        # Summary warnings
        if len(jobs) < len(df):
            self.warnings.append(f"Only {len(jobs)}/{len(df)} rows converted successfully")
        
        return jobs, self.errors, self.warnings
    
    def _extract_job_data(self, row: pd.Series, reverse_map: Dict[str, str]) -> Dict[str, Any]:
        """Extract and convert data for a single job"""
        job_data = {'metadata': {}}
        
        # Required fields
        job_data['job_id'] = self._get_value(row, reverse_map, 'job_id', required=True)
        job_data['processing_time'] = self._get_numeric(row, reverse_map, 'processing_time', required=True)
        
        # Optional core fields
        job_data['operation_id'] = self._get_value(row, reverse_map, 'operation_id')
        job_data['machine'] = self._get_value(row, reverse_map, 'machine')
        
        # Dates
        job_data['due_date'] = self._get_datetime(row, reverse_map, 'due_date')
        job_data['release_date'] = self._get_datetime(row, reverse_map, 'release_date')
        
        # Validate date logic
        if job_data['due_date'] and job_data['release_date']:
            if job_data['due_date'] < job_data['release_date']:
                self.warnings.append(
                    f"Job {job_data['job_id']}: due_date is before release_date"
                )
        
        # Priority
        job_data['priority'] = self._get_value(row, reverse_map, 'priority')
        job_data['priority_numeric'] = self._get_numeric(row, reverse_map, 'priority_numeric')
        
        # Quantities
        job_data['quantity'] = self._get_numeric(row, reverse_map, 'quantity', default=1, as_int=True)
        job_data['lot_size'] = self._get_numeric(row, reverse_map, 'lot_size', as_int=True)
        
        # Outsourcing
        job_data['can_outsource'] = self._get_boolean(row, reverse_map, 'can_outsource', default=False)
        job_data['outsourcing_cost'] = self._get_numeric(row, reverse_map, 'outsourcing_cost')
        job_data['vendor_id'] = self._get_value(row, reverse_map, 'vendor_id')
        
        # Costs and penalties
        job_data['penalty_late'] = self._get_numeric(row, reverse_map, 'penalty_late', default=0)
        job_data['setup_time'] = self._get_numeric(row, reverse_map, 'setup_time', default=0)
        
        # Material/part info
        job_data['part_type'] = self._get_value(row, reverse_map, 'part_type')
        job_data['material_type'] = self._get_value(row, reverse_map, 'material_type')
        job_data['tool_group'] = self._get_value(row, reverse_map, 'tool_group')
        
        # Store any unmapped columns as metadata
        for col in row.index:
            if col not in reverse_map.values() and pd.notna(row[col]):
                job_data['metadata'][col] = self._convert_value(row[col])
        
        return job_data
    
    def _get_value(
        self, 
        row: pd.Series, 
        reverse_map: Dict[str, str], 
        field: str, 
        required: bool = False,
        default: Any = None
    ) -> Any:
        """Get value from row using reverse mapping"""
        if field not in reverse_map:
            if required:
                raise ValueError(f"Required field {field} not found in mapping")
            return default
        
        col = reverse_map[field]
        value = row[col]
        
        if pd.isna(value):
            if required:
                raise ValueError(f"Required field {field} is empty")
            return default
        
        return str(value).strip() if value else default
    
    def _get_numeric(
        self, 
        row: pd.Series, 
        reverse_map: Dict[str, str], 
        field: str, 
        required: bool = False,
        default: Any = None,
        as_int: bool = False
    ) -> Any:
        """Get numeric value from row"""
        if field not in reverse_map:
            if required:
                raise ValueError(f"Required field {field} not found in mapping")
            return default
        
        col = reverse_map[field]
        value = row[col]
        
        if pd.isna(value):
            if required:
                raise ValueError(f"Required field {field} is empty")
            return default
        
        try:
            num_value = float(value)
            return int(num_value) if as_int else num_value
        except (ValueError, TypeError):
            if required:
                raise ValueError(f"Field {field} must be numeric, got: {value}")
            return default
    
    def _get_datetime(
        self, 
        row: pd.Series, 
        reverse_map: Dict[str, str], 
        field: str
    ) -> Any:
        """Get datetime value from row"""
        if field not in reverse_map:
            return None
        
        col = reverse_map[field]
        value = row[col]
        
        if pd.isna(value):
            return None
        
        # If already datetime
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
        
        # Try to parse string
        try:
            return pd.to_datetime(value).to_pydatetime()
        except:
            self.warnings.append(f"Could not parse date value: {value}")
            return None
    
    def _get_boolean(
        self, 
        row: pd.Series, 
        reverse_map: Dict[str, str], 
        field: str,
        default: bool = False
    ) -> bool:
        """Get boolean value from row"""
        if field not in reverse_map:
            return default
        
        col = reverse_map[field]
        value = row[col]
        
        if pd.isna(value):
            return default
        
        # Handle various boolean representations
        if isinstance(value, bool):
            return value
        
        value_str = str(value).lower().strip()
        
        if value_str in ['yes', 'y', 'true', 't', '1', 'x']:
            return True
        elif value_str in ['no', 'n', 'false', 'f', '0', '']:
            return False
        else:
            return default
    
    def _convert_value(self, value: Any) -> Any:
        """Convert pandas value to JSON-serializable type"""
        if pd.isna(value):
            return None
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value
