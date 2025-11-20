# Test Excel Upload Workflow
import requests
import json
import os

API_BASE = "http://localhost:8001"

def test_excel_workflow():
    print("🧪 Testing Excel Upload Workflow...")
    print("=" * 60)
    
    # Step 1: Check if backend is running
    try:
        response = requests.get(f"{API_BASE}/api/data/info")
        print("✅ Backend is running")
    except Exception as e:
        print(f"❌ Backend not running: {e}")
        return
    
    # Step 2: Create a simple test Excel file
    import pandas as pd
    
    test_data = pd.DataFrame({
        'Job_ID': ['J001', 'J002', 'J003'],
        'Operation_ID': ['J001_Op1', 'J002_Op1', 'J003_Op1'],
        'Op_Seq': [1, 1, 1],
        'Quantity': [10, 15, 20],
        'Processing_Time': [30, 45, 60],
        'Setup_Time': [5, 5, 5],
        'Due_Date': [100, 150, 200],
        'Priority': ['HIGH', 'MEDIUM', 'LOW'],
        'Material': ['STEEL', 'ALUMINUM', 'STEEL'],
        'Operation_Type': ['MILLING', 'TURNING', 'MILLING'],
        'Part_Type': ['A', 'B', 'A'],
        'Tool_Group': ['TGA', 'TGB', 'TGA'],
        'Transfer_Min': [5, 5, 5],
        'Release_Day': [0, 0, 0]
    })
    
    test_file = 'test_jobs.xlsx'
    test_data.to_excel(test_file, index=False)
    print(f"✅ Created test file: {test_file}")
    
    # Step 3: Upload Excel
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('test_jobs.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            response = requests.post(f"{API_BASE}/api/excel/upload", files=files)
            
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful: {result.get('sheet_names')}")
            sheet_name = result['sheet_names'][0]
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return
    
    # Step 4: Get auto-mapping
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('test_jobs.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            data = {'sheet_name': sheet_name}
            response = requests.post(
                f"{API_BASE}/api/excel/auto-map",
                files=files,
                data=data
            )
        
        if response.status_code == 200:
            auto_map_result = response.json()
            mappings_list = auto_map_result.get('mappings', [])
            # Convert list to dictionary - mappings should be {excel_column: canonical_field}
            mappings = {item['excel_column']: item['canonical_field'] for item in mappings_list}
            print(f"✅ Auto-mapping created: {mappings}")
        else:
            print(f"⚠️  Auto-mapping failed, using manual mappings")
            mappings = {
                'job_id': 'Job_ID',
                'quantity': 'Quantity',
                'processing_time': 'Processing_Time',
                'setup_time': 'Setup_Time',
                'due_date': 'Due_Date',
                'priority': 'Priority',
                'material_type': 'Material',
                'operation_type': 'Operation_Type'
            }
    except Exception as e:
        print(f"⚠️  Auto-mapping error: {e}, using manual mappings")
        mappings = {
            'job_id': 'Job_ID',
            'quantity': 'Quantity',
            'processing_time': 'Processing_Time',
            'setup_time': 'Setup_Time',
            'due_date': 'Due_Date',
            'priority': 'Priority',
            'material_type': 'Material',
            'operation_type': 'Operation_Type'
        }
    
    # Step 5: Test each heuristic
    heuristics = ['SPT', 'EDD', 'CR', 'PRIORITY']
    
    for heuristic in heuristics:
        print(f"\n📊 Testing {heuristic} heuristic...")
        try:
            with open(test_file, 'rb') as f:
                files = {'file': ('test_jobs.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
                data = {
                    'sheet_name': sheet_name,
                    'mappings': json.dumps(mappings),
                    'heuristic': heuristic
                }
                response = requests.post(
                    f"{API_BASE}/api/excel/load-and-schedule",
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ {heuristic}: Scheduled {result.get('job_count')} jobs")
                print(f"  📈 Metrics: {result.get('metrics')}")
            else:
                print(f"  ❌ {heuristic} failed: {response.status_code}")
                print(f"  Error: {response.text}")
        except Exception as e:
            print(f"  ❌ {heuristic} error: {e}")
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n🧹 Cleaned up test file")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")

if __name__ == "__main__":
    test_excel_workflow()
