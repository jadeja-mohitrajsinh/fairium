import google.generativeai as genai
from typing import Dict, List, Optional
import json
import re
from app.core.config import settings
from app.core.logging import logger

class GeminiAIService:
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not set. Gemini services will be unavailable.")
            self.model = None
            return
            
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI: {e}")
            self.model = None
    
    async def generate_bias_insight(self, metric_data: Dict) -> str:
        """Generate AI-powered insight for bias metrics using Gemini."""
        if not self.model:
            return "AI insight unavailable (Gemini not configured)."
            
        prompt = f"""
        Analyze this bias detection metric and provide a concise, actionable insight:
        Attribute: {metric_data.get('attribute', 'N/A')}
        Disparate Impact Ratio: {metric_data.get('di_ratio', 0):.3f}
        DP Difference: {metric_data.get('dp_diff', 0):.3f}
        Confidence: {metric_data.get('confidence', 'N/A')}
        Severity: {metric_data.get('severity', 'N/A')}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating insight with Gemini: {e}")
            return "Unable to generate AI insight at this time."
    
    async def analyze_text_for_bias(self, text: str) -> Optional[Dict]:
        """Use Gemini to detect subtle tone bias, microaggressions, and stereotypes."""
        if not self.model:
            return None
            
        prompt = f"""
        Analyze the following text for subtle biases, microaggressions, stereotypes, or tone bias.
        Text: "{text}"
        Respond ONLY with a JSON object in the following format:
        {{
            "bias_detected": "Yes" or "No" or "Possible",
            "confidence": "High" or "Medium" or "Low",
            "summary": "Brief summary",
            "biases": [
                {{
                    "type": "Tone/Stereotype/etc",
                    "explanation": "...",
                    "alternatives": ["..."]
                }}
            ]
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            text_response = response.text
            match = re.search(r'```json\n(.*?)\n```', text_response, re.DOTALL)
            if match:
                text_response = match.group(1)
            return json.loads(text_response)
        except Exception as e:
            logger.error(f"Error analyzing text bias with Gemini: {e}")
            return None
