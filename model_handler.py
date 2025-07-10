import os
from typing import Optional

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class ModelHandler:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        """Initialize the model handler with a quantized LLM."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.max_length = 2048
        self.load_model()
    
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
        prompt_parts.append("""You are Ogelo, an intelligent AI assistant with access to documents and the internet. Provide comprehensive, helpful responses by:

1. Using information from provided documents (always cite sources)
2. Adding relevant general knowledge and examples
3. Maintaining conversation context
4. Being factual and acknowledging uncertainty

Example responses:

User: What is renewable energy?
Assistant: Renewable energy comes from natural sources that replenish themselves, like sunlight, wind, and water. For example, solar panels convert sunlight into electricity, while wind turbines harness moving air. The main types include:
- Solar (photovoltaic panels and thermal)
- Wind (onshore and offshore turbines)  
- Hydroelectric (dams and run-of-river)
- Geothermal (earth's heat)
- Biomass (organic materials)

These sources are sustainable because they naturally regenerate, unlike fossil fuels which take millions of years to form.

User: How does machine learning work?
Assistant: Machine learning enables computers to learn patterns from data without explicit programming. Think of it like teaching a child to recognize cats - instead of listing every cat feature, you show many cat photos until they learn the pattern.

The process involves:
1. Training: Feed the algorithm labeled examples
2. Learning: The system identifies patterns in the data
3. Prediction: Apply learned patterns to new, unseen data

For instance, email spam filters learn by analyzing thousands of emails marked as spam or legitimate, then use these patterns to classify new emails automatically.""")
        
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
        """Generate an enhanced rule-based response with context integration."""
        user_lower = user_input.lower()
        
        # Context-aware responses with document integration
        if context and context.strip():
            # Analyze context for key information
            context_snippet = context[:400]
            
            if any(word in user_lower for word in ['what', 'explain', 'describe', 'tell me about']):
                response = f"Based on your documents, I can provide this information: {context_snippet}"
                
                # Add general knowledge enhancement
                if 'definition' in user_lower or 'what is' in user_lower:
                    response += "\n\nTo expand on this, the concept generally involves understanding the relationships between different components and how they work together to achieve specific outcomes."
                
                return response
                
            elif any(word in user_lower for word in ['how', 'why', 'when', 'where']):
                response = f"According to your documents: {context_snippet}"
                
                # Add reasoning and examples
                response += "\n\nThis information helps explain the process. In similar situations, the key factors typically include proper planning, understanding the requirements, and following established best practices."
                
                return response
        
        # Enhanced general responses with examples and context
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm Ogelo, your intelligent RAG assistant. I combine information from your documents with general knowledge to provide comprehensive answers. I can analyze PDFs, text files, CSV data, images (OCR), and web content. What would you like to explore today?"
        
        elif any(word in user_lower for word in ['help', 'what can you do']):
            return """I provide intelligent responses by combining multiple sources:

📚 **Document Analysis**: Search through your uploaded files (PDF, text, CSV, images, web content)
🧠 **Knowledge Integration**: Combine document info with general knowledge and examples  
💬 **Context Awareness**: Remember our conversation for coherent responses
🔍 **Detailed Explanations**: Provide examples, analogies, and step-by-step breakdowns
🌐 **Comprehensive Coverage**: Draw from multiple perspectives to give complete answers

Try asking questions about your documents, or ask me to explain complex topics with examples!"""
        
        elif any(word in user_lower for word in ['thank', 'thanks']):
            return "You're welcome! I'm here to provide comprehensive answers by combining your documents with broader knowledge and examples. Feel free to ask follow-up questions or explore new topics!"
        
        else:
            # Intelligent fallback with knowledge integration
            if context:
                return f"I found relevant information in your documents: {context[:300]}...\n\nCombining this with general knowledge, I can provide additional context and examples to help answer your question comprehensively. Would you like me to elaborate on any specific aspect?"
            else:
                # Provide knowledge-based response even without documents
                response = f"I'd be happy to help with your question about '{user_input}'. "
                
                # Add context based on question type
                if any(word in user_lower for word in ['how', 'explain', 'what']):
                    response += "Let me provide a comprehensive explanation with examples and context. "
                elif any(word in user_lower for word in ['why', 'because']):
                    response += "I can explain the reasoning and provide background context. "
                elif any(word in user_lower for word in ['when', 'where']):
                    response += "I can provide information about timing, location, and relevant circumstances. "
                
                response += "While I don't have specific documents uploaded for this topic yet, I can draw from general knowledge and provide detailed explanations. Would you like me to proceed with a comprehensive answer, or would you prefer to upload relevant documents first?"
                
                return response
    
    def is_model_loaded(self) -> bool:
        """Check if the model is successfully loaded."""
        return self.model is not None and self.tokenizer is not None
