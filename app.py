from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="FortiFund ML Lender Detection Service", version="1.0.0")

# CORS middleware for Supabase edge functions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load MPNet model once at startup (more accurate than MiniLM)
print("Loading MPNet model...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
print("Model loaded successfully!")

# Define known lenders with aliases
# You can expand this list or load from database
lenders = {
    "OnDeck": [
        "ondeck", "ondeck capital", "on deck", "ondeck capital19", "ondeck cap", "on-deck",
        "ondeck fin", "ondeck financing", "on deck financing", "ondeck payment", "ondeck cap fin"
    ],
    "Fundbox": ["fundbox", "fundbox inc", "fundbox inc.", "fund box", "fundbox creditsec"],
    "Revenued": ["revenued", "rev payment", "revenued payment", "revenued pmt"],
    "Kabbage": ["kabbage", "kabb", "k servicing", "kabbage inc"],
    "CAN Capital": ["can capital", "cancapital", "can cap"],
    "BlueVine": ["bluevine", "blue vine", "bluevine capital"],
    "Credibly": ["credibly", "credibly advance"],
    "Forward Financing": ["forward financing", "forward financial"],
    "Rapid Finance": ["rapid finance", "rapid funding"],
    "PayPal Working Capital": ["paypal working capital", "paypal wc", "pp working capital"],
    "Square Capital": ["square capital", "sq capital"],
    "Libertas Funding": ["libertas", "libertas funding"],
    "Greenbox Capital": ["greenbox", "greenbox capital"],
    "Clearco": ["clearco", "clear co", "clearbanc"],
    "Shopify Capital": ["shopify capital", "shopify cap"],
    "Swift Capital": ["swift capital", "swift financial"],
}

# Prepare lender embeddings
lender_names = []
lender_texts = []

for lender, aliases in lenders.items():
    for alias in aliases:
        lender_names.append(lender)
        lender_texts.append(alias)

# Encode lender aliases once at startup
print("Encoding lender aliases...")
lender_embeddings = model.encode(lender_texts, normalize_embeddings=True)
print(f"Encoded {len(lender_texts)} lender aliases")


# Request/Response models
class Transaction(BaseModel):
    date: str
    description: str
    amount: float
    type: str  # 'credit' or 'debit'
    balance: Optional[float] = None


class TransactionRequest(BaseModel):
    transactions: List[Dict]


class DetectedLender(BaseModel):
    lender_name: str
    confidence: float
    status: str  # 'Active' or 'Past'
    first_seen: str
    last_seen: str
    transaction_count: int
    average_amount: float
    total_amount: float
    frequency: str
    chronological_transactions: List[Dict]


class PredictionResponse(BaseModel):
    detected_lenders: List[DetectedLender]
    summary: Dict
    model_info: Dict


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "FortiFund ML Lender Detection",
        "model": "all-mpnet-base-v2",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "lenders_configured": len(lenders),
        "total_aliases": len(lender_texts),
        "model_name": "sentence-transformers/all-mpnet-base-v2"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_lenders(
    request: TransactionRequest, 
    confidence_threshold: float = 0.60,
    min_transactions: int = 4
):
    """
    Analyze transactions and detect MCA lender patterns
    
    Args:
        request: TransactionRequest with list of transactions
        confidence_threshold: Minimum similarity score (default: 0.60, lower for better recall)
        min_transactions: Minimum transactions to identify a position (default: 4)
        
    Returns:
        PredictionResponse with detected lenders and summary
    """
    try:
        transactions = request.transactions
        
        # Filter debit transactions (MCA payments are debits)
        # Only consider debits >= $25 to filter out small charges
        debits = [
            t for t in transactions 
            if t.get('type', '').lower() == 'debit' and t.get('amount', 0) >= 25
        ]
        
        print(f"[DEBUG] Total transactions: {len(transactions)}, Debits >= $25: {len(debits)}")
        
        if len(debits) == 0:
            return PredictionResponse(
                detected_lenders=[],
                summary={
                    'total_positions': 0,
                    'active_positions': 0,
                    'total_daily_obligation': 0.0,
                    'transactions_analyzed': len(transactions),
                    'debit_transactions': 0
                },
                model_info={
                    'model_name': 'all-mpnet-base-v2',
                    'confidence_threshold': confidence_threshold,
                    'min_transactions_for_position': 4
                }
            )
        
        # Extract descriptions
        descriptions = [t.get('description', '') for t in debits]
        print(f"[DEBUG] First 3 descriptions: {descriptions[:3]}")
        
        # Encode transaction descriptions
        txn_embeddings = model.encode(descriptions, normalize_embeddings=True)
        print(f"[DEBUG] Generated embeddings shape: {txn_embeddings.shape}")
        
        # Find best matches for each transaction
        lender_groups = {}
        debug_scores = []
        
        for i, txn in enumerate(debits):
            scores = cosine_similarity([txn_embeddings[i]], lender_embeddings)[0]
            best_idx = scores.argmax()
            score = float(scores[best_idx])
            lender = lender_names[best_idx]
            
            # Log top 3 matches for debugging
            top_3_indices = scores.argsort()[-3:][::-1]
            debug_info = {
                'description': txn.get('description', ''),
                'top_matches': [
                    {'lender': lender_names[idx], 'alias': lender_texts[idx], 'score': round(float(scores[idx]), 3)}
                    for idx in top_3_indices
                ]
            }
            debug_scores.append(debug_info)
            
            # Use configurable confidence threshold
            if score >= confidence_threshold:
                if lender not in lender_groups:
                    lender_groups[lender] = []
                
                lender_groups[lender].append({
                    'date': txn.get('date'),
                    'amount': txn.get('amount'),
                    'description': txn.get('description'),
                    'score': round(score, 3)
                })
        
        # Build detected_lenders response
        detected_lenders = []
        
        print(f"[DEBUG] Lender groups found: {list(lender_groups.keys())}")
        print(f"[DEBUG] Group sizes: {[(k, len(v)) for k, v in lender_groups.items()]}")
        print(f"[DEBUG] Sample debug scores:\n{debug_scores[:5]}")
        
        for lender_name, txns in lender_groups.items():
            # Use configurable minimum transactions
            if len(txns) >= min_transactions:
                txns_sorted = sorted(txns, key=lambda x: x['date'])
                amounts = [t['amount'] for t in txns]
                
                detected_lenders.append({
                    'lender_name': lender_name,
                    'confidence': round(sum(t['score'] for t in txns) / len(txns), 3),
                    'status': 'Active' if is_recent(txns_sorted[-1]['date']) else 'Past',
                    'first_seen': txns_sorted[0]['date'],
                    'last_seen': txns_sorted[-1]['date'],
                    'transaction_count': len(txns),
                    'average_amount': round(sum(amounts) / len(amounts), 2),
                    'total_amount': round(sum(amounts), 2),
                    'frequency': estimate_frequency(txns_sorted),
                    'chronological_transactions': [
                        {
                            'date': t['date'],
                            'amount': t['amount'],
                            'description': t['description']
                        }
                        for t in txns_sorted
                    ]
                })
        
        # Calculate summary
        active_positions = [l for l in detected_lenders if l['status'] == 'Active']
        total_daily_obligation = calculate_daily_obligation(detected_lenders)
        
        print(f"[DEBUG] Final detected lenders: {len(detected_lenders)}")
        
        return PredictionResponse(
            detected_lenders=detected_lenders,
            summary={
                'total_positions': len(detected_lenders),
                'active_positions': len(active_positions),
                'total_daily_obligation': total_daily_obligation,
                'transactions_analyzed': len(transactions),
                'debit_transactions': len(debits)
            },
            model_info={
                'model_name': 'all-mpnet-base-v2',
                'confidence_threshold': confidence_threshold,
                'min_transactions_for_position': min_transactions
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/refresh-lenders")
def refresh_lenders(lenders_data: Dict[str, List[str]]):
    """
    Update known lenders and their aliases dynamically
    Useful when SuperAdmin adds new lenders to the system
    
    Args:
        lenders_data: Dictionary mapping lender names to lists of aliases
        
    Returns:
        Status message
    """
    global lenders, lender_names, lender_texts, lender_embeddings
    
    try:
        lenders = lenders_data
        lender_names = []
        lender_texts = []
        
        for lender, aliases in lenders.items():
            for alias in aliases:
                lender_names.append(lender)
                lender_texts.append(alias)
        
        # Re-encode with new lenders
        lender_embeddings = model.encode(lender_texts, normalize_embeddings=True)
        
        return {
            "status": "success",
            "message": "Lenders updated successfully",
            "total_lenders": len(lenders),
            "total_aliases": len(lender_texts)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update error: {str(e)}")


@app.get("/lenders")
def get_lenders():
    """Get currently configured lenders and their aliases"""
    return {
        "lenders": lenders,
    }



@app.post("/debug-predict")
def debug_predict(request: TransactionRequest):
    """
    Debug endpoint that shows similarity scores for all transactions
    Helps diagnose why lenders aren't being detected
    """
    try:
        transactions = request.transactions
        debits = [
            t for t in transactions 
            if t.get('type', '').lower() == 'debit' and t.get('amount', 0) >= 25
        ]
        
        if len(debits) == 0:
            return {"error": "No debit transactions >= $25 found"}
        
        descriptions = [t.get('description', '') for t in debits]
        txn_embeddings = model.encode(descriptions, normalize_embeddings=True)
        
        results = []
        for i, txn in enumerate(debits):
            scores = cosine_similarity([txn_embeddings[i]], lender_embeddings)[0]
            
            # Get top 5 matches
            top_5_indices = scores.argsort()[-5:][::-1]
            
            results.append({
                'transaction': {
                    'date': txn.get('date'),
                    'description': txn.get('description'),
                    'amount': txn.get('amount')
                },
                'top_matches': [
                    {
                        'rank': idx + 1,
                        'lender': lender_names[top_5_indices[idx]],
                        'alias': lender_texts[top_5_indices[idx]],
                        'score': round(float(scores[top_5_indices[idx]]), 4)
                    }
                    for idx in range(len(top_5_indices))
                ]
            })
        
        return {
            "total_transactions": len(debits),
            "results": results,
            "thresholds": {
                "high_confidence": 0.75,
                "medium_confidence": 0.65,
                "low_confidence": 0.55
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")


# Helper functions
def is_recent(date_str: str) -> bool:
    """Check if transaction is within last 30 days"""
    try:
        txn_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return (datetime.now() - txn_date).days < 30
    except:
        return False


def estimate_frequency(txns: List[Dict]) -> str:
    """Estimate payment frequency based on transaction dates"""
    if len(txns) < 2:
        return 'unknown'
    
    try:
        dates = [datetime.fromisoformat(t['date'].replace('Z', '+00:00')) for t in txns]
        dates.sort()
        
        # Calculate average gap between transactions
        gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        avg_gap = sum(gaps) / len(gaps)
        
        if avg_gap <= 2:
            return 'daily'
        elif avg_gap <= 9:
            return 'weekly'
        elif avg_gap <= 16:
            return 'bi-weekly'
        else:
            return 'monthly'
    except:
        return 'unknown'


def calculate_daily_obligation(lenders: List[Dict]) -> float:
    """Calculate total estimated daily payment obligation"""
    total = 0.0
    
    for lender in lenders:
        if lender['status'] == 'Active':
            avg_amount = lender['average_amount']
            frequency = lender['frequency']
            
            if frequency == 'daily':
                total += avg_amount
            elif frequency == 'weekly':
                total += avg_amount / 7
            elif frequency == 'bi-weekly':
                total += avg_amount / 14
            elif frequency == 'monthly':
                total += avg_amount / 30
            else:
                total += avg_amount / 7  # Default to weekly
    
    return round(total, 2)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
