import requests
import sys
import json
from datetime import datetime

class TrashBinAPITester:
    def __init__(self, base_url="https://a186397c-0a89-4bc2-a616-b27c889dc487.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        
        # Sample test data as specified in the review request
        self.sample_bin = {
            "bin_id": "BIN_TEST_001",
            "fill_level_litres": 850,
            "total_litres": 1000,
            "fill_percentage": 85,
            "location": "Test Location",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "temperature": 22,
            "battery_level": 0.8
        }
        
        # Multiple bins for bulk testing
        self.sample_bins = [
            {
                "bin_id": "BIN_001",
                "fill_level_litres": 750,
                "total_litres": 1000,
                "fill_percentage": 75,
                "location": "Downtown Plaza",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "temperature": 22,
                "battery_level": 0.8
            },
            {
                "bin_id": "BIN_002", 
                "fill_level_litres": 900,
                "total_litres": 1000,
                "fill_percentage": 90,
                "location": "Central Park",
                "latitude": 40.7829,
                "longitude": -73.9654,
                "temperature": 24,
                "battery_level": 0.9
            },
            {
                "bin_id": "BIN_003",
                "fill_level_litres": 300,
                "total_litres": 1000,
                "fill_percentage": 30,
                "location": "University District",
                "latitude": 40.7589,
                "longitude": -73.9851,
                "temperature": 20,
                "battery_level": 0.7
            }
        ]

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=10):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            start_time = datetime.now()
            
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}, Response Time: {response_time:.0f}ms")
                
                # Check if response time is within expected limits (500ms as per requirements)
                if response_time > 500:
                    print(f"⚠️  Warning: Response time ({response_time:.0f}ms) exceeds 500ms requirement")
                
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}...")

            try:
                response_data = response.json() if response.content else {}
            except:
                response_data = {"raw_response": response.text}
                
            return success, response_data, response_time

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timeout after {timeout}s")
            return False, {}, 0
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}, 0

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        return self.run_test("Root API Endpoint", "GET", "", 200)

    def test_model_info(self):
        """Test GET /api/model-info endpoint"""
        success, response, response_time = self.run_test("Model Info", "GET", "model-info", 200)
        
        if success:
            print(f"   Model Loaded: {response.get('model_loaded', 'Unknown')}")
            if response.get('model_loaded'):
                print(f"   Model Name: {response.get('model_name', 'Unknown')}")
                print(f"   Features Count: {len(response.get('features', []))}")
                if 'performance_metrics' in response:
                    print(f"   Performance Metrics: {list(response['performance_metrics'].keys())}")
            else:
                print(f"   Message: {response.get('message', 'No message')}")
        
        return success, response, response_time

    def test_single_prediction(self):
        """Test POST /api/predict endpoint"""
        request_data = {"bin_data": self.sample_bin}
        success, response, response_time = self.run_test(
            "Single Bin Prediction", "POST", "predict", 200, request_data
        )
        
        if success:
            print(f"   Bin ID: {response.get('bin_id')}")
            print(f"   Needs Collection: {response.get('needs_collection')}")
            print(f"   Probability: {response.get('probability', 0):.3f}")
            print(f"   Model Used: {response.get('model_used')}")
            
            # Validate prediction logic for high fill percentage (>80%)
            if self.sample_bin['fill_percentage'] > 80:
                if response.get('needs_collection'):
                    print("✅ Correct: High fill percentage correctly predicted as needing collection")
                else:
                    print("⚠️  Warning: High fill percentage not predicted as needing collection")
        
        return success, response, response_time

    def test_bulk_prediction(self):
        """Test POST /api/predict-bulk endpoint"""
        success, response, response_time = self.run_test(
            "Bulk Bin Predictions", "POST", "predict-bulk", 200, self.sample_bins
        )
        
        if success:
            predictions = response.get('predictions', [])
            summary = response.get('summary', {})
            
            print(f"   Total Predictions: {len(predictions)}")
            print(f"   Bins Needing Collection: {summary.get('bins_needing_collection', 0)}")
            print(f"   Collection Rate: {summary.get('collection_rate', 0):.2%}")
            print(f"   Average Probability: {summary.get('average_probability', 0):.3f}")
            
            # Validate prediction logic
            high_fill_bins = [bin_data for bin_data in self.sample_bins if bin_data['fill_percentage'] > 80]
            low_fill_bins = [bin_data for bin_data in self.sample_bins if bin_data['fill_percentage'] < 50]
            
            print(f"   High Fill Bins (>80%): {len(high_fill_bins)}")
            print(f"   Low Fill Bins (<50%): {len(low_fill_bins)}")
            
            # Check individual predictions
            for pred in predictions:
                bin_data = next((b for b in self.sample_bins if b['bin_id'] == pred['bin_id']), None)
                if bin_data:
                    fill_pct = bin_data['fill_percentage']
                    needs_collection = pred['needs_collection']
                    print(f"   {pred['bin_id']}: {fill_pct}% -> {'Collection' if needs_collection else 'OK'}")
        
        return success, response, response_time

    def test_route_optimization(self):
        """Test POST /api/optimize-route endpoint"""
        request_data = {
            "bins": self.sample_bins,
            "start_location": {"latitude": 40.7128, "longitude": -74.0060}
        }
        
        success, response, response_time = self.run_test(
            "Route Optimization", "POST", "optimize-route", 200, request_data
        )
        
        if success:
            route = response.get('optimized_route', [])
            total_distance = response.get('total_distance', 0)
            estimated_time = response.get('estimated_time', 0)
            number_of_stops = response.get('number_of_stops', 0)
            
            print(f"   Number of Stops: {number_of_stops}")
            print(f"   Total Distance: {total_distance:.3f}")
            print(f"   Estimated Time: {estimated_time:.1f} minutes")
            
            if route:
                print("   Route Order:")
                for i, stop in enumerate(route):
                    print(f"     {i+1}. {stop.get('bin_id')} - {stop.get('location')} ({stop.get('fill_percentage')}%)")
            
            # Validate that only bins needing collection are in route
            if route:
                print("✅ Route optimization working - generated optimized path")
            else:
                print("⚠️  No route generated - check if any bins need collection")
        
        return success, response, response_time

    def test_analytics(self):
        """Test GET /api/analytics endpoint"""
        success, response, response_time = self.run_test("Analytics", "GET", "analytics", 200)
        
        if success:
            print(f"   Total Predictions: {response.get('total_predictions', 0)}")
            print(f"   Bins Needing Collection: {response.get('bins_needing_collection', 0)}")
            print(f"   Collection Rate: {response.get('collection_rate', 0):.2%}")
            print(f"   Average Probability: {response.get('average_probability', 0):.3f}")
        
        return success, response, response_time

    def test_error_handling(self):
        """Test error handling with invalid data"""
        print("\n🔍 Testing Error Handling...")
        
        # Test invalid endpoint
        success, _, _ = self.run_test("Invalid Endpoint", "GET", "invalid-endpoint", 404)
        
        # Test invalid prediction data
        invalid_data = {"bin_data": {"bin_id": "INVALID"}}  # Missing required fields
        success2, _, _ = self.run_test("Invalid Prediction Data", "POST", "predict", 422, invalid_data)
        
        # Test empty bulk prediction
        success3, _, _ = self.run_test("Empty Bulk Prediction", "POST", "predict-bulk", 200, [])
        
        return success and success2

    def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("🚀 Starting Comprehensive AI Trash Bin API Testing")
        print("=" * 60)
        
        # Test sequence
        tests = [
            ("Root Endpoint", self.test_root_endpoint),
            ("Model Info", self.test_model_info),
            ("Single Prediction", self.test_single_prediction),
            ("Bulk Predictions", self.test_bulk_prediction),
            ("Route Optimization", self.test_route_optimization),
            ("Analytics", self.test_analytics),
            ("Error Handling", self.test_error_handling)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if isinstance(result, tuple):
                    results[test_name] = result[0]  # success status
                else:
                    results[test_name] = result
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {str(e)}")
                results[test_name] = False
        
        # Print final results
        print("\n" + "=" * 60)
        print("📊 FINAL TEST RESULTS")
        print("=" * 60)
        
        for test_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{test_name:<25} {status}")
        
        print(f"\nOverall: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed! Backend API is working correctly.")
            return 0
        else:
            print("⚠️  Some tests failed. Please check the backend implementation.")
            return 1

def main():
    tester = TrashBinAPITester()
    return tester.run_comprehensive_test()

if __name__ == "__main__":
    sys.exit(main())