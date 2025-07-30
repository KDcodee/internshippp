from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import json
from scipy.spatial.distance import pdist, squareform

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="AI Trash Bin Management System", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Models for API requests/responses
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class BinData(BaseModel):
    bin_id: str
    fill_level_litres: float
    total_litres: int
    fill_percentage: float
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    temperature: Optional[float] = None
    battery_level: Optional[float] = None
    date: Optional[datetime] = None
    time: Optional[str] = None

class BinPredictionRequest(BaseModel):
    bin_data: BinData

class BinPredictionResponse(BaseModel):
    bin_id: str
    needs_collection: bool
    probability: float
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RouteOptimizationRequest(BaseModel):
    bins: List[BinData]
    start_location: Optional[Dict[str, float]] = None

class RouteOptimizationResponse(BaseModel):
    optimized_route: List[Dict[str, Any]]
    total_distance: float
    estimated_time: Optional[float] = None
    number_of_stops: int

class BulkPredictionResponse(BaseModel):
    predictions: List[BinPredictionResponse]
    summary: Dict[str, Any]

# Global variable to store loaded model
model_artifacts = None

# Utility functions
def load_ml_model():
    """Load the trained ML model artifacts"""
    global model_artifacts
    try:
        model_path = Path(__file__).parent.parent / "trash_bin_model.pkl"
        if model_path.exists():
            model_artifacts = joblib.load(model_path)
            logger.info(f"Model loaded successfully: {model_artifacts['model_name']}")
        else:
            logger.warning("Model file not found. ML predictions will be unavailable.")
            model_artifacts = None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_artifacts = None

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

def extract_temporal_features(date_time: datetime) -> Dict[str, int]:
    """Extract temporal features from datetime"""
    return {
        'day_of_week': date_time.weekday(),
        'month': date_time.month,
        'hour': date_time.hour
    }

def predict_bin_status(bin_data: BinData) -> Dict[str, Any]:
    """Predict if a trash bin needs collection"""
    if model_artifacts is None:
        return {
            'needs_collection': bin_data.fill_percentage > 80,  # Simple fallback
            'probability': min(bin_data.fill_percentage / 100, 1.0),
            'model_used': 'Fallback Rule-based'
        }
    
    # Extract features
    current_time = bin_data.date or datetime.now()
    temporal_features = extract_temporal_features(current_time)
    
    # Prepare feature vector
    features = {
        'FILL_LEVELIN_LITRES': bin_data.fill_level_litres,
        'TOTALLITRES': bin_data.total_litres,
        'FILL_PERCENTAGE': bin_data.fill_percentage,
        'TEMPERATURE_IN_C': bin_data.temperature or 25.0,
        'BATTERY_LEVEL_': bin_data.battery_level or 1.0,
        'day_of_week': temporal_features['day_of_week'],
        'month': temporal_features['month'],
        'hour': temporal_features['hour'],
        'BIN_ID_encoded': hash(bin_data.bin_id) % 100,  # Simple encoding
        'LOCATION_encoded': hash(bin_data.location) % 50  # Simple encoding
    }
    
    # Create feature array
    feature_columns = model_artifacts['feature_columns']
    feature_array = np.array([features.get(col, 0) for col in feature_columns]).reshape(1, -1)
    
    # Make prediction
    model = model_artifacts['model']
    model_name = model_artifacts['model_name']
    
    try:
        if model_name in ['Logistic Regression', 'SVM', 'KNN', 'Naive Bayes']:
            scaler = model_artifacts['scaler']
            feature_array = scaler.transform(feature_array)
        
        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0][1]
        
        return {
            'needs_collection': bool(prediction),
            'probability': float(probability),
            'model_used': model_name
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {
            'needs_collection': bin_data.fill_percentage > 80,
            'probability': min(bin_data.fill_percentage / 100, 1.0),
            'model_used': 'Fallback Rule-based'
        }

def optimize_route(bins: List[BinData], start_location: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Optimize collection route using nearest neighbor heuristic"""
    if not bins:
        return {'route': [], 'total_distance': 0, 'number_of_stops': 0}
    
    # Filter bins with valid coordinates
    valid_bins = [bin_data for bin_data in bins if bin_data.latitude and bin_data.longitude]
    
    if not valid_bins:
        return {'route': [], 'total_distance': 0, 'number_of_stops': 0}
    
    route = []
    total_distance = 0
    unvisited = set(range(len(valid_bins)))
    
    # Start from specified location or first bin
    if start_location and 'latitude' in start_location and 'longitude' in start_location:
        # Find nearest bin to start location
        distances = [
            calculate_distance(
                start_location['latitude'], start_location['longitude'],
                bin_data.latitude, bin_data.longitude
            )
            for bin_data in valid_bins
        ]
        current_idx = distances.index(min(distances))
    else:
        current_idx = 0
    
    # Build route using nearest neighbor
    while unvisited:
        route.append({
            'bin_id': valid_bins[current_idx].bin_id,
            'location': valid_bins[current_idx].location,
            'latitude': valid_bins[current_idx].latitude,
            'longitude': valid_bins[current_idx].longitude,
            'fill_percentage': valid_bins[current_idx].fill_percentage,
            'route_order': len(route) + 1
        })
        
        unvisited.remove(current_idx)
        
        if not unvisited:
            break
        
        # Find nearest unvisited bin
        current_bin = valid_bins[current_idx]
        distances = {}
        
        for idx in unvisited:
            next_bin = valid_bins[idx]
            distance = calculate_distance(
                current_bin.latitude, current_bin.longitude,
                next_bin.latitude, next_bin.longitude
            )
            distances[idx] = distance
        
        next_idx = min(distances, key=distances.get)
        total_distance += distances[next_idx]
        current_idx = next_idx
    
    return {
        'route': route,
        'total_distance': total_distance,
        'number_of_stops': len(route)
    }

# API Routes
@api_router.get("/")
async def root():
    return {"message": "AI Trash Bin Management System API", "version": "1.0.0"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

@api_router.post("/predict", response_model=BinPredictionResponse)
async def predict_bin_collection(request: BinPredictionRequest):
    """Predict if a single bin needs collection"""
    try:
        prediction = predict_bin_status(request.bin_data)
        
        response = BinPredictionResponse(
            bin_id=request.bin_data.bin_id,
            needs_collection=prediction['needs_collection'],
            probability=prediction['probability'],
            model_used=prediction['model_used']
        )
        
        # Store prediction in database
        await db.predictions.insert_one(response.dict())
        
        return response
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.post("/predict-bulk", response_model=BulkPredictionResponse)
async def predict_bulk_bins(bins: List[BinData]):
    """Predict collection needs for multiple bins"""
    try:
        predictions = []
        needs_collection_count = 0
        
        for bin_data in bins:
            prediction = predict_bin_status(bin_data)
            
            response = BinPredictionResponse(
                bin_id=bin_data.bin_id,
                needs_collection=prediction['needs_collection'],
                probability=prediction['probability'],
                model_used=prediction['model_used']
            )
            
            predictions.append(response)
            if prediction['needs_collection']:
                needs_collection_count += 1
        
        # Store predictions in database
        prediction_dicts = [pred.dict() for pred in predictions]
        await db.predictions.insert_many(prediction_dicts)
        
        summary = {
            'total_bins': len(bins),
            'bins_needing_collection': needs_collection_count,
            'collection_rate': needs_collection_count / len(bins) if bins else 0,
            'average_probability': sum(pred.probability for pred in predictions) / len(predictions) if predictions else 0
        }
        
        return BulkPredictionResponse(predictions=predictions, summary=summary)
    except Exception as e:
        logger.error(f"Bulk prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bulk prediction failed: {str(e)}")

@api_router.post("/optimize-route", response_model=RouteOptimizationResponse)
async def optimize_collection_route(request: RouteOptimizationRequest):
    """Optimize collection route for bins that need collection"""
    try:
        # First, predict which bins need collection
        bins_needing_collection = []
        
        for bin_data in request.bins:
            prediction = predict_bin_status(bin_data)
            if prediction['needs_collection']:
                bins_needing_collection.append(bin_data)
        
        if not bins_needing_collection:
            return RouteOptimizationResponse(
                optimized_route=[],
                total_distance=0,
                number_of_stops=0
            )
        
        # Optimize route
        route_result = optimize_route(bins_needing_collection, request.start_location)
        
        # Estimate time (assuming 5 minutes per stop + travel time)
        estimated_time = route_result['number_of_stops'] * 5 + route_result['total_distance'] * 2
        
        response = RouteOptimizationResponse(
            optimized_route=route_result['route'],
            total_distance=route_result['total_distance'],
            estimated_time=estimated_time,
            number_of_stops=route_result['number_of_stops']
        )
        
        # Store route in database
        await db.routes.insert_one({
            'timestamp': datetime.utcnow(),
            'route': route_result['route'],
            'total_distance': route_result['total_distance'],
            'number_of_stops': route_result['number_of_stops']
        })
        
        return response
    except Exception as e:
        logger.error(f"Route optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Route optimization failed: {str(e)}")

@api_router.get("/model-info")
async def get_model_info():
    """Get information about the loaded ML model"""
    if model_artifacts is None:
        return {
            'model_loaded': False,
            'message': 'No ML model loaded. Using fallback rule-based predictions.'
        }
    
    return {
        'model_loaded': True,
        'model_name': model_artifacts['model_name'],
        'features': model_artifacts['feature_columns'],
        'performance_metrics': model_artifacts['performance_metrics']
    }

@api_router.get("/analytics")
async def get_analytics():
    """Get analytics from stored predictions"""
    try:
        total_predictions = await db.predictions.count_documents({})
        
        if total_predictions == 0:
            return {
                'total_predictions': 0,
                'collection_rate': 0,
                'average_probability': 0
            }
        
        # Aggregate analytics
        pipeline = [
            {
                '$group': {
                    '_id': None,
                    'total_predictions': {'$sum': 1},
                    'bins_needing_collection': {
                        '$sum': {'$cond': [{'$eq': ['$needs_collection', True]}, 1, 0]}
                    },
                    'average_probability': {'$avg': '$probability'}
                }
            }
        ]
        
        result = await db.predictions.aggregate(pipeline).to_list(1)
        
        if result:
            stats = result[0]
            return {
                'total_predictions': stats['total_predictions'],
                'bins_needing_collection': stats['bins_needing_collection'],
                'collection_rate': stats['bins_needing_collection'] / stats['total_predictions'],
                'average_probability': stats['average_probability']
            }
        
        return {
            'total_predictions': total_predictions,
            'collection_rate': 0,
            'average_probability': 0
        }
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Load ML model on startup"""
    load_ml_model()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()