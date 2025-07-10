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
    
    def generate_response(self, user_input: str, context: str = None) -> str:
        """Generate a response using the loaded model."""
        if self.model is None or self.tokenizer is None:
            return "Sorry, the model is not available. Please try again later."
        
        # Simple rule-based responses if no ML model is available
        if self.model == "simple":
            return self._simple_response(user_input, context)
        
        try:
            # Construct prompt with context if available
            if context:
                prompt = f"""Context: {context}

User: {user_input}
Assistant: """
            else:
                prompt = f"""User: {user_input}
Assistant: """
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=self.max_length - 200,  # Leave room for generation
                truncation=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=200,
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
            
            # Limit response length
            if len(response) > 500:
                response = response[:500] + "..."
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._simple_response(user_input, context)
    
    def _simple_response(self, user_input: str, context: str = None) -> str:
        """Generate a simple rule-based response."""
        user_lower = user_input.lower()
        
        # Use context if available
        if context and context.strip():
            context_words = context.lower().split()
            input_words = user_lower.split()
            
            # Find overlapping words
            overlap = set(input_words) & set(context_words)
            if overlap:
                return f"Based on the uploaded documents, I found information related to your question about {', '.join(list(overlap)[:3])}. From the context: {context[:200]}..."
        
        # Simple keyword-based responses
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm a RAG chat assistant. I can help answer questions about documents you've uploaded. How can I assist you today?"
        
        elif any(word in user_lower for word in ['what', 'explain', 'tell me']):
            if context:
                return f"Based on the information in your documents: {context[:300]}..."
            else:
                return "I'd be happy to help explain! Please upload some documents first so I can provide relevant information."
        
        elif any(word in user_lower for word in ['how', 'why', 'when', 'where']):
            if context:
                return f"According to your uploaded documents: {context[:300]}..."
            else:
                return "To answer your question, I'll need you to upload some relevant documents first."
        
        elif any(word in user_lower for word in ['thank', 'thanks']):
            return "You're welcome! Feel free to ask me anything about your uploaded documents."
        
        else:
            if context:
                return f"Here's what I found in your documents related to your question: {context[:300]}..."
            else:
                return "I understand your question. Please upload some documents so I can provide you with relevant information from your knowledge base."
    
    def is_model_loaded(self) -> bool:
        """Check if the model is successfully loaded."""
        return self.model is not None and self.tokenizer is not None
