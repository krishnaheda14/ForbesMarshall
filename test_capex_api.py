"""
Quick test script to verify CapEx API endpoints are working
Run this after starting the backend server with: python backend/main.py
"""

import requests
import json

BASE_URL = "http://localhost:8001"

def test_health():
    """Test health check endpoint"""
    print("=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_endpoints_list():
    """Test endpoints listing"""
    print("\n" + "=" * 60)
    print("Testing Endpoints List")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/api/endpoints")
    data = response.json()
    print(f"Status Code: {response.status_code}")
    print(f"Total Endpoints: {data['total_endpoints']}")
    
    # Check if CapEx category exists
    if "CapEx" in data['categories']:
        print("\n✅ CapEx Category Found:")
        for endpoint in data['categories']['CapEx']:
            print(f"  - {endpoint['methods'][0]} {endpoint['path']}")
    else:
        print("\n❌ CapEx category not found!")
    
    return response.status_code == 200

def test_load_data():
    """Load sample data"""
    print("\n" + "=" * 60)
    print("Loading Sample Data")
    print("=" * 60)
    
    response = requests.post(f"{BASE_URL}/api/data/load", json={})
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Data loaded successfully")
        print(f"   Operations: {data.get('operations_count', 0)}")
        print(f"   Machines: {data.get('machines_count', 0)}")
        print(f"   Vendors: {data.get('vendors_count', 0)}")
    else:
        print(f"❌ Failed to load data: {response.text}")
    
    return response.status_code == 200

def test_compute_schedule():
    """Compute a schedule to have some outsourced operations"""
    print("\n" + "=" * 60)
    print("Computing Schedule (SPT)")
    print("=" * 60)
    
    response = requests.post(f"{BASE_URL}/api/schedule/compute", json={
        "heuristic": "SPT",
        "cost_threshold": 0.9
    })
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Schedule computed")
        print(f"   Makespan: {data.get('makespan', 0):.1f}")
        print(f"   Total Tardiness: {data.get('total_tardiness', 0):.1f}")
        print(f"   Outsourced Ops: {data.get('outsourced_count', 0)}")
    else:
        print(f"❌ Failed to compute schedule: {response.text}")
    
    return response.status_code == 200

def test_capex_analyze():
    """Test CapEx analysis endpoint"""
    print("\n" + "=" * 60)
    print("Testing CapEx Analysis")
    print("=" * 60)
    
    response = requests.post(f"{BASE_URL}/api/capex/analyze", params={
        "hourly_labor_rate": 30.0
    })
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ CapEx Analysis Success")
        
        if data.get('biggest_offender'):
            print(f"\n   Biggest Offender: {data['biggest_offender']}")
            print(f"   Outsourced Count: {data['offender_count']}")
            print(f"   Total Vendor Cost: ${data['total_vendor_cost']:,.2f}")
            
            if data.get('recommendations'):
                print(f"\n   Recommendations ({len(data['recommendations'])} machines):")
                for rec in data['recommendations'][:3]:  # Show top 3
                    print(f"\n   📊 Machine: {rec['machine_id']} ({rec['machine_type']})")
                    print(f"      Purchase Price: ${rec['purchase_price']:,.2f}")
                    print(f"      In-House Cost: ${rec['total_inhouse_cost']:,.2f}")
                    print(f"      Savings: ${rec['savings']:,.2f}")
                    if rec['payback_years']:
                        print(f"      Payback: {rec['payback_years']:.1f} years")
                    else:
                        print(f"      Payback: Not profitable")
        else:
            print(f"   {data.get('message', 'No outsourced operations found')}")
    else:
        print(f"❌ Failed: {response.text}")
    
    return response.status_code == 200

def test_capex_buy():
    """Test buying a machine (if recommendations exist)"""
    print("\n" + "=" * 60)
    print("Testing Machine Purchase")
    print("=" * 60)
    
    # First get analysis to find a machine
    analysis = requests.post(f"{BASE_URL}/api/capex/analyze", params={"hourly_labor_rate": 30.0})
    
    if analysis.status_code != 200:
        print("❌ Cannot test buy - analysis failed")
        return False
    
    data = analysis.json()
    
    if not data.get('recommendations') or len(data['recommendations']) == 0:
        print("ℹ️  No machine recommendations available to test purchase")
        return True  # Not a failure, just no data
    
    # Try to buy the first recommended machine
    machine_to_buy = data['recommendations'][0]['machine_id']
    print(f"Attempting to purchase clone of: {machine_to_buy}")
    
    response = requests.post(f"{BASE_URL}/api/capex/buy-machine", json={
        "machine_id": machine_to_buy,
        "hourly_labor_rate": 30.0
    })
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Purchase Successful!")
        print(f"   {result['message']}")
        print(f"   New Machine ID: {result['new_machine_id']}")
        print(f"   Total Machines: {result['machines_count']}")
    else:
        print(f"❌ Failed: {response.text}")
    
    return response.status_code == 200

def main():
    """Run all tests"""
    print("\n🧪 CNC Scheduler - CapEx API Tests")
    print("=" * 60)
    print("Make sure the backend is running on http://localhost:8001")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health),
        ("Endpoints List", test_endpoints_list),
        ("Load Data", test_load_data),
        ("Compute Schedule", test_compute_schedule),
        ("CapEx Analysis", test_capex_analyze),
        ("CapEx Buy Machine", test_capex_buy),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! CapEx feature is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")

if __name__ == "__main__":
    main()
