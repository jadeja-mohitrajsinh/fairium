import re
from typing import Dict, List, Optional
import joblib
from pathlib import Path
from app.core.logging import logger

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
        except Exception as e:
            logger.error(f"Failed to load ML classifier: {e}")
            return None
    
    @classmethod
    def _ml_predict(cls, text: str, model_data: Dict) -> Optional[float]:
        """Get ML prediction for bias probability."""
        try:
            classifier = model_data['classifier']
            vectorizer = model_data['vectorizer']
            text_tfidf = vectorizer.transform([text])
            proba = classifier.predict_proba(text_tfidf)[0]
            bias_probability = proba[1] if len(proba) > 1 else 0.0
            return bias_probability
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return None
    
    @classmethod
    def analyze_text(cls, text: str) -> Dict:
        """Analyze text for potential bias using hybrid approach (rule-based + ML)."""
        logger.info("Analyzing text for potential bias")
        text_lower = text.lower()
        detected_biases = []
        has_ambiguous_bias = False
        
        for bias_type, patterns in cls.BIAS_PATTERNS.items():
            keyword_matches = [kw for kw in patterns['keywords'] if kw in text_lower]
            phrase_matches = [p for p in patterns.get('biased_phrases', []) if re.search(p, text_lower)]
            ambiguous_matches = [p for p in patterns.get('ambiguous_phrases', []) if re.search(p, text_lower)]
            
            if ambiguous_matches:
                has_ambiguous_bias = True
            
            if keyword_matches or phrase_matches or ambiguous_matches:
                match_count = len(keyword_matches) + len(phrase_matches) + len(ambiguous_matches)
                confidence = 'High' if match_count >= 3 else 'Medium' if match_count >= 2 else 'Low'
                
                if ambiguous_matches:
                    explanation = f"Detected {len(ambiguous_matches)} ambiguous phrase(s) related to {bias_type}."
                else:
                    explanation = f"Detected {len(keyword_matches)} keywords and {len(phrase_matches)} biased phrases related to {bias_type}."
                
                alternatives = [f"Replace '{b}' with '{n}'" for b, n in patterns['neutral_alternatives'].items() if b in text_lower]
                
                detected_biases.append({
                    'type': bias_type,
                    'confidence': confidence,
                    'explanation': explanation,
                    'alternatives': alternatives,
                    'keyword_matches': keyword_matches,
                    'phrase_matches': phrase_matches,
                    'ambiguous_matches': ambiguous_matches,
                })
        
        ml_bias_probability = None
        model_data = cls._load_ml_classifier()
        if model_data:
            ml_bias_probability = cls._ml_predict(text, model_data)
        
        if detected_biases:
            confidence_order = {'High': 0, 'Medium': 1, 'Low': 2}
            highest_confidence = min(detected_biases, key=lambda x: confidence_order[x['confidence']])
            bias_status = 'Possible' if has_ambiguous_bias else 'Yes'
            summary = f"Detected {len(detected_biases)} type(s) of potential bias."
            return {
                'bias_detected': bias_status,
                'biases': detected_biases,
                'overall_confidence': highest_confidence['confidence'],
                'ml_confidence': ml_bias_probability,
                'summary': summary
            }
        else:
            if ml_bias_probability and ml_bias_probability > 0.6:
                return {
                    'bias_detected': 'Possible',
                    'biases': [],
                    'overall_confidence': 'Medium',
                    'ml_confidence': ml_bias_probability,
                    'summary': "ML model detected potential bias but no rule-based patterns found."
                }
            else:
                return {
                    'bias_detected': 'No',
                    'biases': [],
                    'overall_confidence': 'Low',
                    'ml_confidence': ml_bias_probability,
                    'summary': 'No bias patterns detected.'
                }
