"""Text bias analysis service for detecting potential bias in text input."""

import re
from typing import Dict, List, Optional
import joblib
from pathlib import Path


class TextBiasAnalyzer:
    """Analyzes text for potential bias, discrimination, or unfair patterns."""
    
    # Bias patterns and keywords
    BIAS_PATTERNS = {
        'gender': {
            'keywords': ['male', 'female', 'man', 'woman', 'men', 'women', 'he', 'she', 'his', 'her', 'him', 'gender', 'sex'],
            'biased_phrases': [
                r'\b(male|female|man|woman)\s+(only|exclusively|must|should|cannot)',
                r'\b(he|she)\s+(is|was|will be)\s+(better|worse|smarter|stronger)',
                r'\b(men|women)\s+(are not|cannot|should not)',
            ],
            'neutral_alternatives': {
                'male only': 'all candidates',
                'female only': 'all candidates',
                'man': 'person',
                'woman': 'person',
                'he': 'they',
                'she': 'they',
            }
        },
        'race': {
            'keywords': ['race', 'ethnicity', 'racial', 'ethnic', 'white', 'black', 'asian', 'hispanic', 'african', 'caucasian'],
            'biased_phrases': [
                r'\b(white|black|asian|hispanic)\s+(only|exclusively|must|should)',
                r'\b(racial|ethnic)\s+(bias|discrimination|preference)',
                r'\b(african|caucasian)\s+(american)',
            ],
            'neutral_alternatives': {
                'white only': 'diverse candidates',
                'black only': 'diverse candidates',
                'racial preference': 'merit-based selection',
            }
        },
        'age': {
            'keywords': ['age', 'young', 'old', 'elderly', 'senior', 'junior', 'youth', 'aged'],
            'biased_phrases': [
                r'\b(young|old)\s+(only|exclusively|must|should)',
                r'\b(elderly|senior)\s+(are not|cannot|too)',
                r'\b(age)\s+(discrimination|bias|requirement)',
            ],
            'neutral_alternatives': {
                'young only': 'all experience levels',
                'old only': 'all experience levels',
                'age requirement': 'experience requirement',
            }
        },
        'socioeconomic': {
            'keywords': ['poor', 'rich', 'wealthy', 'low income', 'high income', 'class', 'economic', 'financial'],
            'biased_phrases': [
                r'\b(poor|low income)\s+(are not|cannot|should not)',
                r'\b(rich|wealthy)\s+(only|exclusively|must)',
                r'\b(economic|financial)\s+(status|background)\s+(requirement)',
            ],
            'neutral_alternatives': {
                'poor': 'candidates from all backgrounds',
                'rich': 'candidates from all backgrounds',
                'low income': 'all economic backgrounds',
            }
        },
        'cultural': {
            'keywords': ['culture', 'cultural', 'fit', 'native', 'traditional', 'mindset'],
            'ambiguous_phrases': [
                r'\btraditional\s+(work|company|office)\s+culture',
                r'\bcultural\s+fit',
                r'\bnative\s+(speaker|english)',
                r'\byoung\s+mindset',
                r'\bculture\s+fit',
                r'\bfit\s+in\s+with\s+our\s+culture',
            ],
            'neutral_alternatives': {
                'traditional work culture': 'collaborative and inclusive work environment',
                'cultural fit': 'alignment with our values',
                'native speaker': 'fluent speaker',
                'young mindset': 'innovative perspective',
            }
        },
        'location': {
            'keywords': ['rural', 'urban', 'city', 'countryside', 'background', 'location', 'regional', 'local'],
            'biased_phrases': [
                r'\bdo\s+not\s+hire\s+(people|candidates)\s+from\s+rural',
                r'\bno\s+(rural|urban|city)\s+(candidates|people)',
                r'\b(rural|urban)\s+(backgrounds|people)\s+not\s+(allowed|welcome)',
                r'\bprefer\s+(urban|city)\s+(candidates|people)',
                r'\blocation\s+(bias|preference|discrimination)',
            ],
            'neutral_alternatives': {
                'rural background': 'candidates from all geographic backgrounds',
                'urban only': 'candidates from all locations',
                'city background': 'candidates from all geographic areas',
            }
        }
    }
    
    @classmethod
    def _load_ml_classifier(cls):
        """Load the ML classifier for bias detection."""
        model_path = Path('data/models/text_bias_classifier.joblib')
        if not model_path.exists():
            return None
        
        try:
            model_data = joblib.load(model_path)
            return model_data
        except Exception:
            return None
    
    @classmethod
    def _ml_predict(cls, text: str, model_data: Dict) -> Optional[float]:
        """Get ML prediction for bias probability."""
        try:
            classifier = model_data['classifier']
            vectorizer = model_data['vectorizer']
            
            # Transform text
            text_tfidf = vectorizer.transform([text])
            
            # Get probability of bias (class 1)
            proba = classifier.predict_proba(text_tfidf)[0]
            bias_probability = proba[1] if len(proba) > 1 else 0.0
            
            return bias_probability
        except Exception:
            return None
    
    @classmethod
    def analyze_text(cls, text: str) -> Dict:
        """
        Analyze text for potential bias using hybrid approach (rule-based + ML).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with bias analysis results
        """
        text_lower = text.lower()
        detected_biases = []
        has_ambiguous_bias = False
        
        # Step 1: Rule-based detection
        for bias_type, patterns in cls.BIAS_PATTERNS.items():
            # Check for keywords
            keyword_matches = []
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    keyword_matches.append(keyword)
            
            # Check for biased phrases using regex
            phrase_matches = []
            if 'biased_phrases' in patterns:
                for pattern in patterns['biased_phrases']:
                    if re.search(pattern, text_lower):
                        phrase_matches.append(pattern)
            
            # Check for ambiguous phrases (vague language that may indicate bias)
            ambiguous_matches = []
            if 'ambiguous_phrases' in patterns:
                for pattern in patterns['ambiguous_phrases']:
                    if re.search(pattern, text_lower):
                        ambiguous_matches.append(pattern)
                        has_ambiguous_bias = True
            
            if keyword_matches or phrase_matches or ambiguous_matches:
                # Determine confidence based on number of matches
                match_count = len(keyword_matches) + len(phrase_matches) + len(ambiguous_matches)
                if match_count >= 3:
                    confidence = 'High'
                elif match_count >= 2:
                    confidence = 'Medium'
                else:
                    confidence = 'Low'
                
                # Generate explanation
                if ambiguous_matches:
                    explanation = f"Detected {len(ambiguous_matches)} ambiguous phrase(s) that may unintentionally exclude individuals from diverse backgrounds related to {bias_type}."
                else:
                    explanation = f"Detected {len(keyword_matches)} bias-related keywords and {len(phrase_matches)} potentially biased phrases related to {bias_type}."
                
                # Suggest neutral alternatives
                alternatives = []
                for biased, neutral in patterns['neutral_alternatives'].items():
                    if biased in text_lower:
                        alternatives.append(f"Replace '{biased}' with '{neutral}'")
                
                detected_biases.append({
                    'type': bias_type,
                    'confidence': confidence,
                    'explanation': explanation,
                    'alternatives': alternatives,
                    'keyword_matches': keyword_matches,
                    'phrase_matches': phrase_matches,
                    'ambiguous_matches': ambiguous_matches,
                })
        
        # Step 2: ML-based detection (hybrid approach)
        ml_bias_probability = None
        model_data = cls._load_ml_classifier()
        if model_data:
            ml_bias_probability = cls._ml_predict(text, model_data)
        
        # Step 3: Combine results
        if detected_biases:
            # Get highest confidence from rule-based
            confidence_order = {'High': 0, 'Medium': 1, 'Low': 2}
            highest_confidence = min(detected_biases, key=lambda x: confidence_order[x['confidence']])
            
            # Determine if bias is definite or possible
            bias_status = 'Possible' if has_ambiguous_bias else 'Yes'
            
            # If ML predicts high bias probability but rules didn't catch it, elevate to Possible
            if ml_bias_probability and ml_bias_probability > 0.7 and bias_status == 'No':
                bias_status = 'Possible'
                summary = f"ML model detected potential bias (confidence: {ml_bias_probability:.2f}). Rule-based analysis found no obvious patterns."
            else:
                summary = f"Detected {len(detected_biases)} type(s) of potential bias: {', '.join([b['type'] for b in detected_biases])}."
                if ml_bias_probability:
                    summary += f" ML confidence: {ml_bias_probability:.2f}"
            
            return {
                'bias_detected': bias_status,
                'biases': detected_biases,
                'overall_confidence': highest_confidence['confidence'],
                'ml_confidence': ml_bias_probability,
                'summary': summary
            }
        else:
            # No rule-based bias detected, check ML
            if ml_bias_probability and ml_bias_probability > 0.6:
                return {
                    'bias_detected': 'Possible',
                    'biases': [],
                    'overall_confidence': 'Medium',
                    'ml_confidence': ml_bias_probability,
                    'summary': f"ML model detected potential bias (confidence: {ml_bias_probability:.2f}) but no obvious rule-based patterns found. Text may contain subtle or context-dependent bias."
                }
            else:
                return {
                    'bias_detected': 'No',
                    'biases': [],
                    'overall_confidence': 'Low',
                    'ml_confidence': ml_bias_probability,
                    'summary': 'No bias patterns detected by either rule-based analysis or ML model.'
                }
