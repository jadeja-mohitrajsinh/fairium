import os
import google.generativeai as genai
from typing import Dict, List

class GeminiAIService:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def generate_bias_insight(self, metric_data: Dict) -> str:
        """Generate AI-powered insight for bias metrics using Gemini."""
        prompt = f"""
        Analyze this bias detection metric and provide a concise, actionable insight:
        
        Attribute: {metric_data.get('attribute', 'N/A')}
        Disparate Impact Ratio: {metric_data.get('di_ratio', 0):.3f}
        DP Difference: {metric_data.get('dp_diff', 0):.3f}
        Confidence: {metric_data.get('confidence', 'N/A')}
        Severity: {metric_data.get('severity', 'N/A')}
        
        Provide a 1-2 sentence insight explaining:
        1. What this means for fairness
        2. Why it matters
        3. What action should be taken
        
        Keep it concise and business-focused.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating insight with Gemini: {e}")
            return "Unable to generate AI insight at this time."
    
    async def generate_recommendation(self, bias_summary: Dict) -> str:
        """Generate AI-powered recommendation based on overall bias analysis."""
        prompt = f"""
        Based on this bias analysis summary, provide a concise recommendation:
        
        Total Records: {bias_summary.get('total_records', 0)}
        Risk Level: {bias_summary.get('risk_level', 'N/A')}
        Critical Issues: {len(bias_summary.get('critical_metrics', []))}
        
        Provide a 1-2 sentence recommendation for:
        1. Immediate action
        2. Long-term improvement
        
        Keep it actionable and specific.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating recommendation with Gemini: {e}")
            return "Review bias metrics and implement fairness-aware model training."
            
    async def analyze_text_for_bias(self, text: str) -> Dict:
        """Use Gemini to detect subtle tone bias, microaggressions, and stereotypes."""
        prompt = f"""
        Analyze the following text for subtle biases, microaggressions, stereotypes, or tone bias.
        
        Text: "{text}"
        
        Respond ONLY with a JSON object in the following format:
        {{
            "bias_detected": "Yes" or "No" or "Possible",
            "confidence": "High" or "Medium" or "Low",
            "summary": "Brief summary of the bias found or why it's neutral",
            "biases": [
                {{
                    "type": "Tone/Stereotype/Microaggression/etc",
                    "explanation": "Why this is biased",
                    "alternatives": ["Alternative phrasing 1", "Alternative phrasing 2"]
                }}
            ]
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            import json
            import re
            
            # Extract JSON block if surrounded by markdown
            text_response = response.text
            match = re.search(r'```json\n(.*?)\n```', text_response, re.DOTALL)
            if match:
                text_response = match.group(1)
                
            return json.loads(text_response)
        except Exception as e:
            print(f"Error analyzing text bias with Gemini: {e}")
            return None
