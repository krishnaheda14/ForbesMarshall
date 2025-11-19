# backend/models.py
"""
Canonical scheduling schema models
Framework-independent representation of scheduling entities
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class PriorityClass(str, Enum):
    """Priority classification"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    A = "A"
    B = "B"
    C = "C"


class CanonicalJob(BaseModel):
    """
    Canonical internal representation of a job/operation
    Industry-agnostic schema that all uploaded data maps to
    """
    # Core identifiers
    job_id: str = Field(..., description="Unique job identifier")
    operation_id: Optional[str] = Field(None, description="Operation ID for multi-stage jobs")
    
    # Scheduling parameters
    machine: Optional[str] = Field(None, description="Assigned machine/work center")
    processing_time: float = Field(..., gt=0, description="Processing time in hours")
    
    # Time constraints
    due_date: Optional[datetime] = Field(None, description="Job due date")
    release_date: Optional[datetime] = Field(None, description="Earliest start date")
    
    # Priority and classification
    priority: Optional[str] = Field(None, description="Priority class (A/B/C or HIGH/MEDIUM/LOW)")
    priority_numeric: Optional[int] = Field(None, ge=1, le=10, description="Numeric priority (1-10)")
    
    # Production details
    quantity: Optional[int] = Field(1, ge=1, description="Lot size or quantity")
    lot_size: Optional[int] = Field(None, ge=1, description="Batch size")
    
    # Outsourcing
    can_outsource: bool = Field(False, description="Whether this job can be outsourced")
    outsourcing_cost: Optional[float] = Field(None, ge=0, description="Cost to outsource")
    vendor_id: Optional[str] = Field(None, description="Preferred vendor if outsourced")
    
    # Penalties and costs
    penalty_late: Optional[float] = Field(0, ge=0, description="Penalty per time unit if late")
    setup_time: Optional[float] = Field(0, ge=0, description="Setup time in hours")
    
    # Material/part details
    part_type: Optional[str] = Field(None, description="Part or product type")
    material_type: Optional[str] = Field(None, description="Material type (for setup penalties)")
    tool_group: Optional[str] = Field(None, description="Required tool group")
    
    # Additional metadata (industry-specific)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extra fields")
    
    # Internal scheduling state
    assigned_machine: Optional[str] = Field(None, description="Scheduled machine")
    scheduled_start: Optional[datetime] = Field(None, description="Scheduled start time")
    scheduled_end: Optional[datetime] = Field(None, description="Scheduled end time")
    is_outsourced: bool = Field(False, description="Whether scheduled as outsourced")
    tardiness: Optional[float] = Field(None, description="Lateness in hours")
    
    @validator('due_date', 'release_date')
    def validate_dates(cls, v, values):
        """Ensure dates are valid"""
        if v and isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except:
                return None
        return v
    
    @validator('processing_time')
    def validate_processing_time(cls, v):
        """Ensure processing time is positive"""
        if v <= 0:
            raise ValueError("Processing time must be positive")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class CanonicalMachine(BaseModel):
    """Canonical machine/resource representation"""
    machine_id: str
    machine_type: Optional[str] = None
    capacity: Optional[float] = None  # Operations per day, etc.
    available_from: Optional[datetime] = None
    available_to: Optional[datetime] = None
    hourly_cost: Optional[float] = None
    
    # Capabilities
    supported_operations: Optional[List[str]] = Field(default_factory=list)
    tool_groups: Optional[List[str]] = Field(default_factory=list)
    
    # Maintenance windows
    maintenance_windows: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    
    # Performance factors
    speed_factor: float = Field(1.0, gt=0, description="Speed multiplier (1.0 = normal)")
    oee: Optional[float] = Field(None, ge=0, le=1, description="Overall Equipment Effectiveness")
    
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ColumnMapping(BaseModel):
    """Mapping of Excel columns to canonical schema fields"""
    excel_column: str
    canonical_field: str  # Field name in CanonicalJob
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    source: str = Field(description="'heuristic', 'llm', or 'user'")
    data_type: Optional[str] = Field(None, description="Detected data type")
    sample_values: Optional[List[Any]] = Field(None, description="Sample values from column")


class MappingTemplate(BaseModel):
    """Reusable mapping template for specific customer/industry"""
    template_name: str
    description: Optional[str] = None
    industry: Optional[str] = None
    mappings: List[ColumnMapping]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ScheduleConfig(BaseModel):
    """Configuration for running scheduling algorithms"""
    heuristic: str = Field(..., description="SPT, EDD, CR, PRIORITY, etc.")
    outsourcing_threshold: float = Field(0.9, ge=0, description="Max cost multiplier for outsourcing")
    current_time: Optional[datetime] = Field(None, description="Reference time for scheduling")
    consider_release_dates: bool = Field(True)
    consider_priorities: bool = Field(True)
    machine_ids: Optional[List[str]] = Field(None, description="Machines to include")
    
    # Machine breakdowns to simulate
    breakdowns: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
