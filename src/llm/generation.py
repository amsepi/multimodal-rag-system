from openai import OpenAI
from typing import List, Dict
import streamlit as st
import re
import os
from src.config.settings import settings

class ResponseGenerator:
    def __init__(self):
        # Get API key from Streamlit secrets or environment
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
        )
        
    def optimize_context(self, context: List[Dict], max_tokens: int = None) -> str:
        """Optimize context to fit within token limits while preserving important information"""
        if not context:
            return ""
        
        # Use settings if not provided
        config = settings.get_llm_config()
        max_tokens = max_tokens or config["context_max_tokens"]
        
        # Sort by relevance score
        sorted_context = sorted(context, key=lambda x: x.get('score', 0), reverse=True)
        
        # Start with highest scoring chunks
        optimized_chunks = []
        current_length = 0
        
        for chunk in sorted_context:
            chunk_text = f"Source (Page {chunk['metadata']['page']}): {chunk['content']}"
            chunk_length = len(chunk_text)
            
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = chunk_length // 4
            
            if current_length + estimated_tokens <= max_tokens:
                optimized_chunks.append(chunk_text)
                current_length += estimated_tokens
            else:
                # If we can't fit the full chunk, try to fit a truncated version
                remaining_tokens = max_tokens - current_length
                remaining_chars = remaining_tokens * 4
                
                if remaining_chars > 100:  # Only if we have meaningful space left
                    truncated_text = chunk['content'][:remaining_chars] + "..."
                    optimized_chunks.append(f"Source (Page {chunk['metadata']['page']}): {truncated_text}")
                break
        
        return "\n\n".join(optimized_chunks)
    
    def create_enhanced_prompt(self, query: str, context: str, strategy: str = "comprehensive") -> str:
        """Create enhanced prompts for better answer generation"""
        
        base_prompts = {
            "comprehensive": f"""You are a helpful assistant analyzing university documents. Use the following sources to answer the question comprehensively.

SOURCES:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Analyze ALL provided sources carefully
2. Extract specific facts, numbers, and details from the sources
3. If the sources contain conflicting information, mention this
4. If information is missing from the sources, state this clearly
5. Provide a complete, well-structured answer that addresses all aspects of the question
6. Include relevant page numbers and source references
7. If the question asks for specific numbers, dates, or requirements, provide them exactly as stated in the sources

ANSWER:""",

            "step_by_step": f"""Analyze the following sources step-by-step to answer the question.

SOURCES:
{context}

QUESTION: {query}

STEPS:
1. First, identify what specific information is being requested
2. Search through the sources for relevant facts and details
3. Extract any numbers, dates, requirements, or specific criteria mentioned
4. Organize the information logically
5. Synthesize a comprehensive answer that addresses all parts of the question

ANSWER:""",

            "detailed": f"""You are an expert document analyst. Provide a detailed, comprehensive answer based on the following sources.

SOURCES:
{context}

QUESTION: {query}

REQUIREMENTS:
- Be thorough and complete in your response
- Include specific details, numbers, and dates when available
- Reference the source pages where information comes from
- If the question asks for requirements, list them completely
- If the question asks for financial information, provide exact figures
- If information is incomplete, acknowledge what's missing
- Structure your answer clearly with proper formatting

DETAILED ANSWER:""",

            "academic": f"""As an academic document assistant, provide a scholarly analysis of the following sources.

SOURCES:
{context}

QUESTION: {query}

ANALYSIS FRAMEWORK:
1. Context: What is the broader context of this information?
2. Key Findings: What are the main points from the sources?
3. Specific Details: What exact numbers, dates, or requirements are mentioned?
4. Implications: What does this information mean for the question asked?
5. Completeness: Are there any gaps in the information provided?

SCHOLARLY ANSWER:"""
        }
        
        return base_prompts.get(strategy, base_prompts["comprehensive"])
    
    def generate_response(self, query: str, context: List[Dict], strategy: str=None) -> str:
        """Generate comprehensive answer using enhanced prompting"""
        
        if not context:
            return "I couldn't find any relevant information in the documents to answer your question."
        
        # Use settings if not provided
        config = settings.get_llm_config()
        strategy = strategy or config["default_strategy"]
        
        # Optimize context to fit within token limits
        optimized_context = self.optimize_context(context)
        
        # Create enhanced prompt
        prompt = self.create_enhanced_prompt(query, optimized_context, strategy)
        
        try:
            response = self.client.chat.completions.create(
                model=config["model"],
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            # Post-process the answer to ensure completeness
            answer = self.post_process_answer(answer, query, context)
            
            return answer
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def post_process_answer(self, answer: str, query: str, context: List[Dict]) -> str:
        """Post-process answer to ensure completeness and quality"""
        
        # Check if answer is too short for a comprehensive question
        if len(answer.strip()) < settings.MIN_ANSWER_LENGTH and any(word in query.lower() for word in ['what', 'how', 'why', 'explain', 'describe']):
            # Try to generate a more detailed answer
            return self.generate_response(query, context, "detailed")
        
        # Check if answer mentions missing information
        if "I couldn't find" in answer or "no information" in answer.lower():
            # Try with more context or different strategy
            if len(context) < 5:
                return self.generate_response(query, context, "step_by_step")
        
        return answer
    
    def generate_response_with_fallback(self, query: str, context: List[Dict]) -> str:
        """Generate response with multiple fallback strategies"""
        
        config = settings.get_llm_config()
        strategies = config["fallback_strategies"]
        
        for strategy in strategies:
            try:
                response = self.generate_response(query, context, strategy)
                
                # Check if response is satisfactory
                if len(response.strip()) > 50 and not response.startswith("Error"):
                    return response
                    
            except Exception as e:
                print(f"Strategy {strategy} failed: {str(e)}")
                continue
        
        # Final fallback
        return "I apologize, but I'm unable to generate a comprehensive answer at this time. Please try rephrasing your question or check if the relevant documents are available."