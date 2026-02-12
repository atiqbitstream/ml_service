from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

app = FastAPI(title="FortiFund ML Lender Detection Service (Optimized)", version="2.0.0")

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

# Define known lenders with aliases (EXPANDED for better matching)
lenders = {
    "OnDeck": [
        "ondeck", "ondeck capital", "on deck", "ondeck capital19", "ondeck cap", "on-deck",
        "ondeck fin", "ondeck financing", "on deck financing", "ondeck payment", "ondeck cap fin",
        "ondeck capital 19", "on deck cap"
    ],
    "Fundbox": [
        "fundbox", "fundbox inc", "fundbox inc.", "fund box", "fundbox creditsec",
        "fundbox inc creditsec", "fbxinc", "fundbox adv"
    ],
    "Revenued": [
        "revenued", "rev payment", "revenued payment", "revenued pmt", "revenued debit",
        "revenued processing"
    ],
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
    "American Express": ["american express", "amex", "amex loan", "american express loan"],
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


# ============================================================================
# ADVANCED TEXT PREPROCESSING FOR NOISY ACH DESCRIPTIONS
# ============================================================================

def extract_company_name_from_ach(description: str) -> str:
    """
    Extract clean company name from noisy ACH transaction descriptions.
    Handles formats like: "Orig CO Name:Fundbox Inc. Orig ID:Fbxinc Desc Date:..."
    """
    if not description:
        return ""
    
    # Try to extract "Orig CO Name:" field (most reliable)
    ach_pattern = r"Orig CO Name:\s*([^O]+?)(?:\s+Orig\s+|$)"
    match = re.search(ach_pattern, description, re.IGNORECASE)
    if match:
        company = match.group(1).strip()
        # Remove common suffixes that don't help matching
        company = re.sub(r'\s+(Inc\.?|LLC|Corp\.?|Ltd\.?|Co\.?)\s*$', '', company, flags=re.IGNORECASE)
        return company
    
    return description


def normalize_description(description: str) -> str:
    """
    Normalize transaction description by removing common noise tokens.
    This dramatically improves matching accuracy on noisy data.
    """
    if not description:
        return ""
    
    # Extract company name if it's ACH format
    clean_text = extract_company_name_from_ach(description)
    
    # If extraction failed, use original
    if not clean_text or clean_text == description:
        clean_text = description
    
    # Remove common ACH noise tokens
    noise_patterns = [
        r'Orig\s+(?:CO\s+)?Name:',
        r'Orig\s+ID:[^\s]*',
        r'Desc\s+Date:[^\s]*',
        r'CO\s+Entry\s+Descr:',
        r'Sec:[A-Z]+',
        r'Trace#:[^\s]+',
        r'Eed:[^\s]+',
        r'Ind\s+(?:ID|Name):[^T]+',
        r'Trn:[^\s]+Tc?',
        r'Collect:[^\s]+',
        r'\d{10,}',  # Long numbers (likely IDs)
    ]
    
    for pattern in noise_patterns:
        clean_text = re.sub(pattern, ' ', clean_text, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    clean_text = ' '.join(clean_text.split())
    
    return clean_text.strip()


def fuzzy_match_score(text1: str, text2: str) -> float:
    """Calculate fuzzy string matching score (0-1)"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def keyword_match(description: str, lender_alias: str) -> bool:
    """Check if lender keyword appears in description (case-insensitive)"""
    if not description or not lender_alias:
        return False
    
    desc_lower = description.lower()
    alias_lower = lender_alias.lower()
    
    # Direct substring match
    if alias_lower in desc_lower:
        return True
    
    # Word boundary match (more precise)
    words = desc_lower.split()
    alias_words = alias_lower.split()
    
    # Check if all alias words appear in description
    return all(any(aw in w for w in words) for aw in alias_words)


def multi_strategy_match(description: str, lender_alias: str, semantic_score: float) -> Tuple[float, str]:
    """
    Use multiple strategies to match lender:
    1. Semantic similarity (mpNET model)
    2. Keyword matching (exact substring)
    3. Fuzzy matching (for typos/variants)
    
    Returns: (combined_score, match_method)
    """
    # Strategy 1: Semantic similarity from mpNET
    semantic_weight = semantic_score
    
    # Strategy 2: Keyword matching (boost if exact match found)
    keyword_boost = 0.0
    if keyword_match(description, lender_alias):
        keyword_boost = 0.40  # Strong boost for exact keyword match
    
    # Strategy 3: Fuzzy matching (for close variants)
    fuzzy_score = fuzzy_match_score(
        normalize_description(description),
        lender_alias
    )
    fuzzy_weight = fuzzy_score * 0.20  # Moderate weight
    
    # Combine scores
    combined_score = semantic_weight + keyword_boost + fuzzy_weight
    combined_score = min(combined_score, 1.0)  # Cap at 1.0
    
    # Determine match method
    if keyword_boost > 0:
        method = "keyword+semantic"
    elif fuzzy_score > 0.8:
        method = "fuzzy+semantic"
    else:
        method = "semantic_only"
    
    return combined_score, method


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_recent(date_str: str, days_threshold: int = 90) -> bool:
    """Check if a transaction date is recent (within last 90 days)"""
    try:
        txn_date = datetime.strptime(date_str, '%Y-%m-%d')
        return (datetime.now() - txn_date).days <= days_threshold
    except:
        return False


def estimate_frequency(transactions: List[Dict]) -> str:
    """Estimate payment frequency based on transaction dates"""
    if len(transactions) < 2:
        return "Unknown"
    
    dates = sorted([datetime.strptime(t['date'], '%Y-%m-%d') for t in transactions])
    intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    avg_interval = sum(intervals) / len(intervals) if intervals else 0
    
    if avg_interval <= 1.5:
        return "Daily"
    elif avg_interval <= 7:
        return "Weekly"
    elif avg_interval <= 10:
        return "Bi-weekly"
    elif avg_interval <= 35:
        return "Monthly"
    else:
        return "Irregular"


def calculate_daily_obligation(detected_lenders: List[Dict]) -> float:
    """Calculate total estimated daily payment obligation"""
    daily_total = 0.0
    
    for lender in detected_lenders:
        if lender['status'] == 'Active':
            freq = lender['frequency']
            avg_amount = lender['average_amount']
            
            # Convert to daily equivalent
            if freq == "Daily":
                daily_total += avg_amount
            elif freq == "Weekly":
                daily_total += avg_amount / 5  # Business days
            elif freq == "Bi-weekly":
                daily_total += avg_amount / 10
            elif freq == "Monthly":
                daily_total += avg_amount / 22  # ~22 business days/month
    
    return round(daily_total, 2)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TransactionRequest(BaseModel):
    transactions: List[Dict]


class PredictionResponse(BaseModel):
    detected_lenders: List[Dict]
    summary: Dict
    model_info: Dict


# ============================================================================
# MAIN PREDICTION ENDPOINT (OPTIMIZED)
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
def predict_lenders(
    request: TransactionRequest, 
    confidence_threshold: float = 0.45,  # LOWERED from 0.60 for better recall on noisy data
    min_transactions: int = 3  # LOWERED from 4 to catch more patterns
):
    """
    Analyze transactions and detect MCA lender patterns (OPTIMIZED VERSION)
    
    Args:
        request: TransactionRequest with list of transactions
        confidence_threshold: Minimum similarity score (default: 0.45, optimized for noisy ACH data)
        min_transactions: Minimum transactions to identify a position (default: 3)
        
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
                    'model_name': 'all-mpnet-base-v2-optimized',
                    'confidence_threshold': confidence_threshold,
                    'min_transactions_for_position': min_transactions,
                    'optimizations_applied': [
                        'ACH description preprocessing',
                        'Multi-strategy matching (semantic+keyword+fuzzy)',
                        'Lowered confidence threshold',
                        'Enhanced normalization'
                    ]
                }
            )
        
        # ============================================
        # OPTIMIZATION: Preprocess descriptions
        # ============================================
        raw_descriptions = [t.get('description', '') for t in debits]
        normalized_descriptions = [normalize_description(desc) for desc in raw_descriptions]
        
        print(f"[DEBUG] First 3 raw descriptions: {raw_descriptions[:3]}")
        print(f"[DEBUG] First 3 normalized: {normalized_descriptions[:3]}")
        
        # Encode NORMALIZED descriptions (cleaner embeddings)
        txn_embeddings = model.encode(normalized_descriptions, normalize_embeddings=True)
        print(f"[DEBUG] Generated embeddings shape: {txn_embeddings.shape}")
        
        # Find best matches for each transaction using MULTI-STRATEGY matching
        lender_groups = {}
        debug_scores = []
        
        for i, txn in enumerate(debits):
            raw_desc = raw_descriptions[i]
            norm_desc = normalized_descriptions[i]
            
            # Get semantic similarity scores
            semantic_scores = cosine_similarity([txn_embeddings[i]], lender_embeddings)[0]
            
            # Apply multi-strategy matching for each lender alias
            best_score = 0.0
            best_lender = None
            best_method = ""
            
            for idx, (lender_name, lender_alias) in enumerate(zip(lender_names, lender_texts)):
                semantic_score = float(semantic_scores[idx])
                
                # Use multi-strategy matching (semantic + keyword + fuzzy)
                combined_score, match_method = multi_strategy_match(
                    raw_desc, lender_alias, semantic_score
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_lender = lender_name
                    best_method = match_method
            
            # Log top 3 matches for debugging
            top_3_indices = semantic_scores.argsort()[-3:][::-1]
            debug_info = {
                'raw_description': raw_desc[:80],  # Truncate for readability
                'normalized': norm_desc,
                'best_match': {
                    'lender': best_lender,
                    'score': round(best_score, 3),
                    'method': best_method
                },
                'top_semantic_matches': [
                    {
                        'lender': lender_names[idx],
                        'alias': lender_texts[idx],
                        'semantic_score': round(float(semantic_scores[idx]), 3)
                    }
                    for idx in top_3_indices
                ]
            }
            debug_scores.append(debug_info)
            
            # Use LOWERED confidence threshold with multi-strategy score
            if best_score >= confidence_threshold and best_lender:
                if best_lender not in lender_groups:
                    lender_groups[best_lender] = []
                
                lender_groups[best_lender].append({
                    'date': txn.get('date'),
                    'amount': txn.get('amount'),
                    'description': raw_desc,
                    'normalized_description': norm_desc,
                    'score': round(best_score, 3),
                    'match_method': best_method
                })
        
        # Build detected_lenders response
        detected_lenders = []
        
        print(f"[DEBUG] Lender groups found: {list(lender_groups.keys())}")
        print(f"[DEBUG] Group sizes: {[(k, len(v)) for k, v in lender_groups.items()]}")
        print(f"[DEBUG] Sample debug scores (first 3):")
        for debug_info in debug_scores[:3]:
            print(f"  - {debug_info}")
        
        for lender_name, txns in lender_groups.items():
            # Use configurable minimum transactions (LOWERED to 3)
            if len(txns) >= min_transactions:
                txns_sorted = sorted(txns, key=lambda x: x['date'])
                amounts = [t['amount'] for t in txns]
                
                # Calculate average confidence and check match methods
                avg_confidence = sum(t['score'] for t in txns) / len(txns)
                match_methods = [t.get('match_method', 'semantic_only') for t in txns]
                primary_method = max(set(match_methods), key=match_methods.count)
                
                print(f"[DEBUG] Detected: {lender_name} - {len(txns)} txns, avg_conf={avg_confidence:.3f}, method={primary_method}")
                
                detected_lenders.append({
                    'lender_name': lender_name,
                    'confidence': round(avg_confidence, 3),
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
                'model_name': 'all-mpnet-base-v2-optimized',
                'confidence_threshold': confidence_threshold,
                'min_transactions_for_position': min_transactions,
                'optimizations_applied': [
                    'ACH description preprocessing',
                    'Multi-strategy matching (semantic+keyword+fuzzy)',
                    'Lowered confidence threshold',
                    'Enhanced normalization'
                ]
            }
        )
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[ERROR] Prediction failed: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================================
# DEBUG ENDPOINT (OPTIMIZED)
# ============================================================================

@app.post("/debug-predict")
def debug_predict(request: TransactionRequest, confidence_threshold: float = 0.45):
    """
    Debug endpoint - returns detailed matching scores for analysis (OPTIMIZED)
    """
    try:
        transactions = request.transactions
        debits = [
            t for t in transactions 
            if t.get('type', '').lower() == 'debit' and t.get('amount', 0) >= 25
        ]
        
        if len(debits) == 0:
            return {"message": "No qualifying debit transactions found", "results": []}
        
        # Preprocess descriptions
        raw_descriptions = [t.get('description', '') for t in debits]
        normalized_descriptions = [normalize_description(desc) for desc in raw_descriptions]
        
        # Encode normalized descriptions
        txn_embeddings = model.encode(normalized_descriptions, normalize_embeddings=True)
        
        results = []
        for i, txn in enumerate(debits):
            raw_desc = raw_descriptions[i]
            norm_desc = normalized_descriptions[i]
            
            # Get semantic scores
            semantic_scores = cosine_similarity([txn_embeddings[i]], lender_embeddings)[0]
            
            # Apply multi-strategy matching
            all_matches = []
            for idx, (lender_name, lender_alias) in enumerate(zip(lender_names, lender_texts)):
                semantic_score = float(semantic_scores[idx])
                combined_score, match_method = multi_strategy_match(
                    raw_desc, lender_alias, semantic_score
                )
                
                all_matches.append({
                    'lender': lender_name,
                    'alias': lender_alias,
                    'semantic_score': round(semantic_score, 3),
                    'combined_score': round(combined_score, 3),
                    'match_method': match_method,
                    'above_threshold': combined_score >= confidence_threshold
                })
            
            # Sort by combined score
            all_matches.sort(key=lambda x: x['combined_score'], reverse=True)
            
            results.append({
                'date': txn.get('date'),
                'amount': txn.get('amount'),
                'raw_description': raw_desc,
                'normalized_description': norm_desc,
                'top_10_matches': all_matches[:10],
                'matches_above_threshold': len([m for m in all_matches if m['above_threshold']])
            })
        
        return {
            'transactions_analyzed': len(debits),
            'confidence_threshold': confidence_threshold,
            'optimizations_enabled': True,
            'results': results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug prediction failed: {str(e)}")


# ============================================================================
# HEALTH CHECK & INFO ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "service": "FortiFund ML Lender Detection (Optimized)",
        "version": "2.0.0",
        "model": "all-mpnet-base-v2",
        "optimizations": [
            "ACH description preprocessing with regex extraction",
            "Multi-strategy matching (semantic + keyword + fuzzy)",
            "Lowered confidence threshold (0.45 default)",
            "Enhanced text normalization",
            "Expanded lender alias database"
        ],
        "endpoints": {
            "/predict": "Main prediction endpoint (POST)",
            "/debug-predict": "Debug endpoint with detailed scores (POST)",
            "/health": "Health check endpoint (GET)",
            "/lenders": "List all known lenders and aliases (GET)"
        },
        "status": "ready"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "embeddings_ready": lender_embeddings is not None,
        "lender_count": len(lenders),
        "alias_count": len(lender_texts)
    }


@app.get("/lenders")
def get_lenders():
    return {
        "lenders": lenders,
        "total_lenders": len(lenders),
        "total_aliases": len(lender_texts)
    }
