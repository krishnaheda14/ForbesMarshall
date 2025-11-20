# backend/schema_mapping.py
"""
Automatic schema understanding and mapping
Combines heuristic rules + LLM reasoning to map Excel columns to canonical schema
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import re
import google.generativeai as genai
import json
import requests
import os


class SchemaMapper:
    """
    Maps arbitrary Excel columns to canonical scheduling schema
    Uses both rule-based heuristics and LLM reasoning
    """
    
    # Define all possible canonical field mappings
    CANONICAL_FIELDS = {
        'job_id': 'Unique job or order identifier',
        'operation_id': 'Operation ID for multi-stage jobs',
        'machine': 'Machine, work center, or resource ID',
        'processing_time': 'Processing or runtime in hours/minutes',
        'due_date': 'Job due date or deadline',
        'release_date': 'Earliest start date or release date',
        'priority': 'Priority class (A/B/C or HIGH/MEDIUM/LOW)',
        'priority_numeric': 'Numeric priority value (1-10)',
        'quantity': 'Lot size, batch size, or quantity',
        'can_outsource': 'Whether job can be outsourced (Yes/No)',
        'outsourcing_cost': 'Cost to outsource this job',
        'vendor_id': 'Preferred vendor or supplier',
        'penalty_late': 'Penalty for late completion',
        'setup_time': 'Setup or changeover time',
        'part_type': 'Part number, product type, or SKU',
        'material_type': 'Material type or grade',
        'tool_group': 'Required tool or tooling group',
        'customer': 'Customer name (metadata)',
        'ignore': 'Column to skip/ignore'
    }
    
    def __init__(self, gemini_model=None, openrouter_api_key=None, use_openrouter=False):
        """Initialize with optional Gemini AI model or OpenRouter API key"""
        self.gemini_model = gemini_model
        self.openrouter_api_key = openrouter_api_key
        self.use_openrouter = use_openrouter
        
    def map_heuristic(self, columns_info: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Rule-based heuristic mapping using column names and data types
        
        Args:
            columns_info: List of column metadata from ExcelIngestor
            
        Returns:
            Dict mapping column names to {field, confidence, reasoning}
        """
        mappings = {}
        
        for col_info in columns_info:
            col_name = col_info['column_name']
            inferred = col_info.get('inferred_type', 'other')
            
            # Start with inferred type
            if inferred != 'other':
                confidence = 0.7  # Medium confidence for heuristic
                
                # Boost confidence for strong matches
                col_lower = col_name.lower()
                
                if inferred == 'job_id':
                    if 'id' in col_lower or 'number' in col_lower:
                        confidence = 0.9
                elif inferred == 'processing_time':
                    if col_info['is_numeric'] and ('time' in col_lower or 'duration' in col_lower):
                        confidence = 0.85
                elif inferred == 'due_date' or inferred == 'release_date':
                    if col_info['is_datetime'] or 'date' in col_lower:
                        confidence = 0.9
                elif inferred == 'machine':
                    if 'machine' in col_lower or 'resource' in col_lower:
                        confidence = 0.85
                elif inferred == 'priority':
                    if 'priority' in col_lower or 'class' in col_lower:
                        confidence = 0.9
                
                mappings[col_name] = {
                    'canonical_field': inferred,
                    'confidence': confidence,
                    'source': 'heuristic',
                    'reasoning': f"Column name pattern suggests {inferred}"
                }
            else:
                # Low confidence - mark as other/ignore
                mappings[col_name] = {
                    'canonical_field': 'ignore',
                    'confidence': 0.3,
                    'source': 'heuristic',
                    'reasoning': "No clear pattern match"
                }
        
        return mappings
    
    def map_llm(
        self, 
        columns_info: List[Dict[str, Any]], 
        sample_rows: Optional[List[Dict]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        LLM-based mapping using OpenRouter (Claude) or Gemini AI
        
        Args:
            columns_info: Column metadata
            sample_rows: Sample data rows for context
            
        Returns:
            Dict mapping column names to {field, confidence, reasoning}
        """
        # Check if we have either API available
        if not self.use_openrouter and not self.gemini_model:
            print("[SchemaMapper] No AI model available, skipping LLM mapping")
            return {}
        
        try:
            # Build prompt
            prompt = self._build_llm_prompt(columns_info, sample_rows)
            
            # Use OpenRouter (Claude) if available and preferred
            if self.use_openrouter and self.openrouter_api_key:
                print("[SchemaMapper] Calling OpenRouter (Claude 3.5 Sonnet) for column mapping...")
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "anthropic/claude-3.5-sonnet",
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 2000,
                        "temperature": 0.3
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result['choices'][0]['message']['content'].strip()
                else:
                    # Fallback to Gemini if OpenRouter fails
                    if self.gemini_model:
                        print(f"[SchemaMapper] OpenRouter failed ({response.status_code}), trying Gemini...")
                        response_obj = self.gemini_model.generate_content(prompt)
                        response_text = response_obj.text.strip()
                    else:
                        print(f"[SchemaMapper] OpenRouter API error: {response.status_code} - {response.text}")
                        return {}
            else:
                # Use Gemini
                print("[SchemaMapper] Calling Gemini API for column mapping...")
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text.strip()
            
            # Extract JSON from markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(response_text)
            
            # Convert to our format
            mappings = {}
            for col_name, mapping_data in result.get('column_mappings', {}).items():
                if isinstance(mapping_data, str):
                    # Simple string response
                    canonical_field = mapping_data
                    confidence = 0.8
                    reasoning = "LLM suggestion"
                else:
                    # Detailed response
                    canonical_field = mapping_data.get('field', 'ignore')
                    confidence = mapping_data.get('confidence', 0.8)
                    reasoning = mapping_data.get('reasoning', 'LLM suggestion')
                
                mappings[col_name] = {
                    'canonical_field': canonical_field,
                    'confidence': confidence,
                    'source': 'llm',
                    'reasoning': reasoning
                }
            
            print(f"[SchemaMapper] LLM successfully mapped {len(mappings)} columns")
            return mappings
            
        except Exception as e:
            print(f"[SchemaMapper] LLM mapping failed: {type(e).__name__}: {str(e)}")
            print(f"[SchemaMapper] Falling back to heuristic-only mapping")
            return {}
    
    def _build_llm_prompt(
        self, 
        columns_info: List[Dict[str, Any]], 
        sample_rows: Optional[List[Dict]] = None
    ) -> str:
        """Build prompt for LLM to understand schema"""
        
        # Field descriptions
        field_descriptions = "\n".join([
            f"- {field}: {desc}" 
            for field, desc in self.CANONICAL_FIELDS.items()
        ])
        
        # Column information
        column_details = []
        for col_info in columns_info:
            col_name = col_info['column_name']
            samples = col_info.get('sample_values', [])[:3]
            dtype = col_info.get('data_type', 'unknown')
            
            column_details.append(
                f"  - '{col_name}': type={dtype}, samples={samples}"
            )
        
        columns_str = "\n".join(column_details)
        
        # Sample rows if provided
        sample_data_str = ""
        if sample_rows and len(sample_rows) > 0:
            sample_data_str = "\n\nSample rows (first 3):\n"
            for i, row in enumerate(sample_rows[:3], 1):
                sample_data_str += f"Row {i}: {json.dumps(row, default=str)}\n"
        
        prompt = f"""You are an expert in job scheduling and manufacturing data analysis.

I have an Excel file with scheduling data that needs to be mapped to a canonical schema for a scheduling system.

**Available canonical fields:**
{field_descriptions}

**Excel columns found:**
{columns_str}
{sample_data_str}

**Task:**
Map each Excel column to ONE of the canonical fields listed above. If a column doesn't fit any field, map it to 'ignore'.

Consider:
1. Column names and their semantic meaning
2. Data types (numeric, date, text)
3. Sample values provided
4. Common industry terminology

**Required output format (strict JSON):**
{{
  "column_mappings": {{
    "Excel Column Name": {{
      "field": "canonical_field_name",
      "confidence": 0.9,
      "reasoning": "Brief explanation"
    }},
    ...
  }}
}}

Provide ONLY the JSON response, no additional text."""

        return prompt
    
    def combine_mappings(
        self, 
        heuristic_mappings: Dict[str, Dict], 
        llm_mappings: Dict[str, Dict]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Combine heuristic and LLM mappings, taking the higher confidence one
        
        Returns:
            Final mappings with source indicating which method was used
        """
        combined = {}
        
        all_columns = set(heuristic_mappings.keys()) | set(llm_mappings.keys())
        
        for col in all_columns:
            heur = heuristic_mappings.get(col, {})
            llm = llm_mappings.get(col, {})
            
            heur_conf = heur.get('confidence', 0)
            llm_conf = llm.get('confidence', 0)
            
            if llm_conf > heur_conf:
                combined[col] = llm
                combined[col]['source'] = 'llm+heuristic' if heur_conf > 0 else 'llm'
            else:
                combined[col] = heur
                combined[col]['source'] = 'heuristic+llm' if llm_conf > 0 else 'heuristic'
        
        return combined
    
    def auto_map(
        self, 
        columns_info: List[Dict[str, Any]], 
        sample_rows: Optional[List[Dict]] = None,
        use_llm: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Automatic mapping using both methods
        
        Returns:
            Combined mappings with confidence scores
        """
        # Get heuristic mappings
        heuristic = self.map_heuristic(columns_info)
        
        # Get LLM mappings if enabled
        llm = {}
        if use_llm and self.gemini_model:
            llm = self.map_llm(columns_info, sample_rows)
        
        # Combine
        if llm:
            return self.combine_mappings(heuristic, llm)
        else:
            return heuristic
