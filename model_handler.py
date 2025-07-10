import os
from typing import Optional

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import web search capability
try:
    from perplexity_search import PerplexitySearch
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False

class ModelHandler:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        """Initialize the model handler with a quantized LLM."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.max_length = 2048
        
        # Initialize web search capability
        if PERPLEXITY_AVAILABLE:
            self.web_search = PerplexitySearch()
        else:
            self.web_search = None
        self.available_models = {
            "microsoft/Phi-3-mini-4k-instruct": {
                "name": "Phi-3 Mini (Recommended)",
                "description": "Microsoft's efficient 4K context model, optimized for CPU",
                "max_length": 4096
            },
            "microsoft/DialoGPT-medium": {
                "name": "DialoGPT Medium",
                "description": "Conversational AI model, good for dialogue",
                "max_length": 1024
            }
        }
        self.load_model()
    
    def get_available_models(self):
        """Get list of available models with descriptions."""
        return self.available_models
    
    def switch_model(self, new_model_name: str):
        """Switch to a different model."""
        if new_model_name in self.available_models:
            self.model_name = new_model_name
            self.max_length = self.available_models[new_model_name]["max_length"]
            self.load_model()
            return True
        return False
    
    def load_model(self):
        """Load the quantized model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers library not available - using simple rule-based responses")
            self.model = "simple"
            self.tokenizer = "simple"
            return
            
        try:
            print(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with CPU optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to simple rule-based responses")
            self.model = "simple"
            self.tokenizer = "simple"
    
    def generate_response(self, user_input: str, context: str = None, conversation_history: list = None) -> str:
        """Generate a response using advanced prompt engineering and context integration."""
        if self.model is None or self.tokenizer is None:
            return "Sorry, the model is not available. Please try again later."
        
        # Check for real-time information needs FIRST, before any other processing
        real_time_indicators = ['latest', 'recent', 'current', 'news', 'today', 'now', 'update', 'breaking', '2024', '2025', 'weather', 'forecast', 'happened', 'developments']
        needs_real_time = any(indicator in user_input.lower() for indicator in real_time_indicators)
        
        # Try web search FIRST for real-time questions, regardless of context or model type
        if needs_real_time and self.web_search and self.web_search.is_available():
            web_answer, web_sources = self.web_search.enhanced_search(user_input, context)
            if web_answer and len(web_answer) > 100 and "Web search" not in web_answer and "failed" not in web_answer.lower():
                # Format with web sources
                if web_sources:
                    web_answer += f"\n\n**Web Sources:**\n"
                    for i, source in enumerate(web_sources[:3], 1):
                        web_answer += f"{i}. {source}\n"
                return web_answer
        
        # Enhanced rule-based responses if no ML model is available
        if self.model == "simple":
            return self._enhanced_simple_response(user_input, context, conversation_history)
        
        try:
            # Build enhanced prompt with examples and context
            prompt = self._build_enhanced_prompt(user_input, context, conversation_history)
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=self.max_length - 300,  # Leave room for generation
                truncation=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=250,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated part
            response = full_response[len(prompt):].strip()
            
            # Clean up response
            if not response:
                response = "I understand your question, but I'm having trouble generating a response right now. Could you please rephrase your question?"
            
            # Enhance response with additional context if needed
            response = self._enhance_response_with_knowledge(response, user_input)
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._enhanced_simple_response(user_input, context, conversation_history)
    
    def _build_enhanced_prompt(self, user_input: str, context: str = None, conversation_history: list = None) -> str:
        """Build an enhanced prompt with examples and structured context."""
        prompt_parts = []
        
        # System instruction with examples
        prompt_parts.append("""You are Ogelo, an intelligent AI assistant with access to documents and knowledge bases. Provide comprehensive, helpful responses by:

1. Using information from provided documents (always cite sources)
2. Adding relevant general knowledge and examples with references
3. Maintaining conversation context
4. Being factual and acknowledging uncertainty
5. Always including source citations and references for every answer

Example responses with proper citations:

User: What is renewable energy?
Assistant: Renewable energy comes from natural sources that replenish themselves, like sunlight, wind, and water. For example, solar panels convert sunlight into electricity, while wind turbines harness moving air. The main types include:
- Solar (photovoltaic panels and thermal)
- Wind (onshore and offshore turbines)  
- Hydroelectric (dams and run-of-river)
- Geothermal (earth's heat)
- Biomass (organic materials)

These sources are sustainable because they naturally regenerate, unlike fossil fuels which take millions of years to form.

**References:** Environmental science knowledge base, energy industry standards

User: How does machine learning work?
Assistant: Machine learning enables computers to learn patterns from data without explicit programming. Think of it like teaching a child to recognize cats - instead of listing every cat feature, you show many cat photos until they learn the pattern.

The process involves:
1. Training: Feed the algorithm labeled examples
2. Learning: The system identifies patterns in the data
3. Prediction: Apply learned patterns to new, unseen data

For instance, email spam filters learn by analyzing thousands of emails marked as spam or legitimate, then use these patterns to classify new emails automatically.

**References:** Computer science fundamentals, machine learning textbooks, practical AI applications""")
        
        # Add conversation history for context
        if conversation_history and len(conversation_history) > 0:
            prompt_parts.append("\nRecent conversation:")
            for msg in conversation_history[-2:]:  # Last 2 exchanges
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:150]  # Truncate long messages
                if role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
        
        # Add document context if available
        if context and context.strip():
            prompt_parts.append(f"\nDocument information:\n{context[:800]}")
        
        # Add the current question
        prompt_parts.append(f"\nUser: {user_input}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _enhance_response_with_knowledge(self, response: str, user_input: str) -> str:
        """Enhance response with additional knowledge and examples."""
        user_lower = user_input.lower()
        
        # Add examples for certain topics
        if any(term in user_lower for term in ['explain', 'what is', 'how does']):
            if len(response) < 100:  # Short response, add more context
                if 'example' not in response.lower():
                    response += "\n\nFor example, " + self._get_contextual_example(user_input)
        
        return response
    
    def _get_contextual_example(self, user_input: str) -> str:
        """Get a contextual example based on the user's question."""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['technology', 'computer', 'software']):
            return "in software development, this principle is often applied in creating user interfaces that adapt to user behavior."
        elif any(word in user_lower for word in ['science', 'research', 'study']):
            return "researchers often use this approach to validate their hypotheses through controlled experiments."
        elif any(word in user_lower for word in ['business', 'company', 'market']):
            return "companies like Amazon and Google have successfully implemented similar strategies to improve customer experience."
        else:
            return "this concept applies to many real-world situations where understanding patterns and relationships is important."
    
    def _enhanced_simple_response(self, user_input: str, context: str = None, conversation_history: list = None) -> str:
        """Generate an enhanced rule-based response with context integration and proper citations."""
        user_lower = user_input.lower()
        
        # Enhanced context-aware responses that directly answer questions
        if context and context.strip():
            # Check if context contains useful document information (not just conversation history)
            has_document_info = any(indicator in context for indicator in ["From ", "documents", "uploaded", "file"])
            has_conversation_only = "Previous topic:" in context and not has_document_info
            
            # If context is just old conversations, ignore it and provide knowledge response
            if has_conversation_only or len(context) < 100:
                from web_search_integration import WebSearchIntegrator
                integrator = WebSearchIntegrator()
                enhanced_context = integrator._enhance_with_knowledge(user_input)
                if enhanced_context and len(enhanced_context) > 100:
                    return enhanced_context
            
            # Extract knowledge from context to provide direct answers
            from web_search_integration import WebSearchIntegrator
            integrator = WebSearchIntegrator()
            
            # If context already contains enhanced knowledge, use it directly  
            if any(marker in context for marker in ["**Machine Learning", "**Software Development", "**Climate Change", "**Renewable Energy", "**Data Science", "**Business &", "**Health &", "**Education &"]):
                return context
            
            # Otherwise, enhance the context and provide a direct answer
            enhanced_context = integrator._enhance_with_knowledge(user_input, context)
            if enhanced_context and len(enhanced_context) > 100:
                return enhanced_context
            
            # Fallback to basic context processing
            context_snippet = context[:500]
            
            if any(word in user_lower for word in ['what', 'explain', 'describe', 'tell me about']):
                response = f"Based on available information: {context_snippet}"
                
                # Add knowledge enhancement
                response += "\n\n**Additional Context:**"
                response += "\nThis concept involves understanding key relationships and principles that apply across different situations."
                response += "\n\n**Sources:** Available documents and knowledge base"
                
                return response
                
            elif any(word in user_lower for word in ['how', 'why', 'when', 'where']):
                response = f"**Answer:** {context_snippet}"
                
                # Add practical guidance
                response += "\n\n**Key Points:**"
                response += "\nThe process typically involves systematic steps, proper methodology, and attention to important details."
                response += "\n\n**Sources:** Document analysis and established practices"
                
                return response
        
        # Check for real-time information needs first
        real_time_indicators = ['latest', 'recent', 'current', 'news', 'today', 'now', 'update', 'breaking', '2024', '2025', 'weather', 'forecast', 'happened', 'developments']
        needs_real_time = any(indicator in user_lower for indicator in real_time_indicators)
        
        # Try web search first for real-time questions
        if needs_real_time and self.web_search and self.web_search.is_available():
            web_answer, web_sources = self.web_search.enhanced_search(user_input, context)
            if web_answer and len(web_answer) > 100:
                # Format with web sources
                if web_sources:
                    web_answer += f"\n\n**Web Sources:**\n"
                    for i, source in enumerate(web_sources[:3], 1):
                        web_answer += f"{i}. {source}\n"
                return web_answer
        
        # Check for knowledge questions (after real-time check)
        if any(word in user_lower for word in ['what is', 'what are', 'how does', 'how do', 'explain', 'tell me about', 'describe']):
            # Try web search first for real-time information
            if self.web_search and self.web_search.is_available():
                web_answer, web_sources = self.web_search.enhanced_search(user_input, context)
                if web_answer and len(web_answer) > 100:
                    # Format with web sources
                    if web_sources:
                        web_answer += f"\n\n**Web Sources:**\n"
                        for i, source in enumerate(web_sources[:3], 1):
                            web_answer += f"{i}. {source}\n"
                    return web_answer
            
            # Fallback to internal knowledge
            from web_search_integration import WebSearchIntegrator
            integrator = WebSearchIntegrator()
            enhanced_context = integrator._enhance_with_knowledge(user_input)
            
            if enhanced_context and len(enhanced_context) > 100:
                return enhanced_context
        
        # Enhanced general responses with examples and context  
        if any(word in user_lower for word in ['hello', 'hi', 'hey']) and not any(word in user_lower for word in ['what', 'how', 'explain']):
            return "Hello! I'm Ogelo, your intelligent RAG assistant. I can provide comprehensive answers by combining information from your documents with my extensive knowledge base. I can analyze PDFs, text files, CSV data, images (OCR), and web content. What would you like to explore today?"
        
        elif any(word in user_lower for word in ['help', 'what can you do']):
            return """I provide intelligent responses by combining multiple sources with proper citations:

📚 **Document Analysis**: Search through your uploaded files (PDF, text, CSV, images, web content)
🧠 **Knowledge Integration**: Combine document info with my knowledge base and web research capabilities
💬 **Context Awareness**: Remember our conversation for coherent responses
🔍 **Detailed Explanations**: Provide examples, analogies, and step-by-step breakdowns with references
🌐 **Web Research**: Access current information and provide source citations
📖 **Citation System**: Every answer includes clear references to sources used

**How I Reference Information:**
- Document sources: Citations from your uploaded files
- Knowledge base: References to established knowledge and best practices
- Web research: Links and sources from current information (when available)
- Conversation context: References to our previous discussion

Try asking questions about your documents, or ask me to explain complex topics with proper citations!"""
        
        elif any(word in user_lower for word in ['thank', 'thanks']):
            return "You're welcome! I'm here to provide comprehensive answers by combining your documents with broader knowledge and examples. Feel free to ask follow-up questions or explore new topics!"
        
        else:
            # Try web search for general questions
            if self.web_search and self.web_search.is_available():
                web_answer, web_sources = self.web_search.enhanced_search(user_input, context)
                if web_answer and len(web_answer) > 50:
                    if web_sources:
                        web_answer += f"\n\n**Web Sources:**\n"
                        for i, source in enumerate(web_sources[:3], 1):
                            web_answer += f"{i}. {source}\n"
                    return web_answer
            
            # Intelligent fallback with knowledge integration and citations
            if context:
                response = f"**From Your Documents:** {context[:300]}..."
                response += "\n\n**Knowledge Integration:** Combining this with my knowledge base, I can provide additional context and examples to help answer your question comprehensively."
                response += "\n\n**References:**"
                response += "\n- Your uploaded documents"
                response += "\n- General knowledge synthesis"
                response += "\n- Contextual analysis"
                response += "\n\nWould you like me to elaborate on any specific aspect?"
                return response
            else:
                # Get enhanced knowledge from web search integration
                from web_search_integration import WebSearchIntegrator
                integrator = WebSearchIntegrator()
                enhanced_context = integrator._enhance_with_knowledge(user_input)
                
                if enhanced_context and len(enhanced_context) > 100:
                    # Return the comprehensive knowledge response directly
                    return enhanced_context
                else:
                    # Fallback for edge cases
                    response = f"**Knowledge-Based Response for: '{user_input}'**\n\n"
                    
                    # Add context based on question type
                    if any(word in user_lower for word in ['how', 'explain', 'what']):
                        response += "I can provide a comprehensive explanation with examples and context from my knowledge base. "
                    elif any(word in user_lower for word in ['why', 'because']):
                        response += "I can explain the reasoning and provide background context from established knowledge. "
                    elif any(word in user_lower for word in ['when', 'where']):
                        response += "I can provide information about timing, location, and relevant circumstances from available knowledge. "
                    
                    response += "\n\n**Available Knowledge Sources:**"
                    response += "\n- Internal knowledge base"
                    response += "\n- General domain expertise"
                    response += "\n- Best practices and principles"
                    response += "\n- Contextual examples and analogies"
                    
                    response += "\n\nI have extensive knowledge across many domains including science, technology, business, health, education, and more. Would you like me to proceed with a comprehensive answer, or would you prefer to upload relevant documents for more specific information?"
                
                return response
    
    def is_model_loaded(self) -> bool:
        """Check if the model is successfully loaded."""
        return self.model is not None and self.tokenizer is not None
    
    def is_web_search_available(self) -> bool:
        """Check if web search functionality is available."""
        return self.web_search is not None and self.web_search.is_available()
    
    def test_web_search(self) -> tuple:
        """Test web search connection."""
        if self.web_search:
            return self.web_search.test_connection()
        return False, "Web search not initialized"
