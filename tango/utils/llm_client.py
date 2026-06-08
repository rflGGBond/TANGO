import json
import os
from typing import Dict, Any, Optional
import warnings
import logging
import openai
import torch
import ast

# Completely suppress Hugging Face transformers warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # Try to import torch and transformers for local model support
# try:
    
#     _LOCAL_DEPS_AVAILABLE = True
# except ImportError:
#     _LOCAL_DEPS_AVAILABLE = False

class LLMClient:
    """
    A simple client to interact with an LLM provider (e.g., OpenAI, Anthropic, or Local).
    Supports local models deployed in a specific directory.
    """
    def __init__(self, provider: str = "local", api_key: Optional[str] = None, model: str = "Qwen2.5-14B", model_root: str = "../../../models", base_url: Optional[str] = None):
        
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.model_root = model_root
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        self.pipeline = None
        
        if self.provider == "openai" and not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when provider='openai'.")
        
        if self.provider == "local":
            self._init_local_model()

    def _init_local_model(self):
        """Initialize the local model and tokenizer."""
        model_path = os.path.join(self.model_root, self.model)
        if not os.path.exists(model_path):
            # Try to see if the user provided a full path or a relative path that exists
            if os.path.exists(self.model):
                model_path = self.model
            else:
                raise FileNotFoundError(f"Model path not found: {model_path}")
        
        print(f"Loading local model from {model_path}...")
        try:
            # Use device_map="auto" to handle large models if accelerate is installed
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Ensure pad_token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model_obj = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="balanced_low_0", 
                dtype="auto", 
                trust_remote_code=True
            )
            
            # Set seed for reproducibility if torch is available
            if "torch" in globals():
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_obj,
                tokenizer=self.tokenizer,
                device_map="balanced_low_0" # Ensure pipeline handles devices correctly
            )
            print("Local model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")

    def get_completion(self, system_prompt: str, user_prompt: str, response_format: str = "json", temperature: float = 0.5) -> str:
        """
        Sends a prompt to the LLM and returns the response content.
        """
        if self.provider == "mock":
            return self._mock_response(system_prompt, user_prompt)
        
        elif self.provider == "local":
            response = self._local_response(system_prompt, user_prompt, response_format, temperature)
            if response_format == "json":
                return self._clean_and_extract_json(response)
            return response

        elif self.provider == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError("Import openai Error.")

            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            kwargs = {
                "model": self.model,
                "messages": messages,
            }
            
            if response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}
            
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if response_format == "json":
                return self._clean_and_extract_json(content)
            return content
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _clean_and_extract_json(self, text: str) -> str:
        """
        Extracts JSON substring from text and attempts to fix common issues.
        """
        # Debug print to see what LLM is actually returning
        # print(f"DEBUG: Raw LLM response: {text[:200]}..." if len(text) > 200 else f"DEBUG: Raw LLM response: {text}")

        original_text = text
        
        # Remove Markdown code blocks if present
        extracted_from_markdown = None
        if "```json" in text:
            try:
                extracted_from_markdown = text.split("```json")[1].split("```")[0]
            except IndexError:
                pass
        elif "```" in text:
            try:
                extracted_from_markdown = text.split("```")[1].split("```")[0]
            except IndexError:
                pass
        
        # If we successfully extracted something AND it looks like it has JSON, use it.
        # Otherwise, stick to the original text (or maybe the text before the markdown?)
        
        candidate_text = text
        if extracted_from_markdown and "{" in extracted_from_markdown:
             candidate_text = extracted_from_markdown
        
        text = candidate_text.strip()
        
        # Find the first '{' and the last '}'
        start = text.find("{")
        end = text.rfind("}")
        
        # Handle Truncated JSON (missing closing brace)
        if start != -1 and end == -1:
            # Try appending '}' or '"}' to fix common truncation
            for suffix in ["}", "\"}"]:
                try:
                    temp_text = text[start:] + suffix
                    json.loads(temp_text)
                    return temp_text
                except:
                    pass
        if start != -1 and end != -1:
            extracted_text = text[start:end+1]
            try:
                json.loads(extracted_text)
                return extracted_text
            except json.JSONDecodeError as e:
                # Handle "Extra data" error by truncating at the error position
                if e.msg.startswith("Extra data"):
                    try:
                        truncated_text = extracted_text[:e.pos]
                        json.loads(truncated_text)
                        return truncated_text
                    except:
                        pass
                
                # Handle comments (//) which strict JSON doesn't allow
                try:
                    import re
                    no_comments = re.sub(r'(?<!:)\/\/.*', '', extracted_text)
                    json.loads(no_comments)
                    return no_comments
                except:
                    pass
                pass
                
            # 2. Try ast.literal_eval (handles Python-style dicts with single quotes, etc.)
            try:
                # ast.literal_eval is safe for literal structures
                py_obj = ast.literal_eval(extracted_text)
                return json.dumps(py_obj)
            except (ValueError, SyntaxError):
                pass
                
            # 3. Simple manual fixes (last resort)
            try:
                fixed_text = extracted_text.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
                json.loads(fixed_text)
                return fixed_text
            except json.JSONDecodeError:
                pass
                
        # 4. Aggressive truncation fix for arrays (e.g. [1, 2, 3...)
        if start != -1:
            raw_text = text[start:]
            # If it's missing closing braces/brackets, let's try to forcefully close them
            # Very common in LLMs: "candidate_seed_set": [226, 27, 262...
            if raw_text.count("[") > raw_text.count("]"):
                # Cut off at the last comma to remove incomplete numbers
                last_comma = raw_text.rfind(",")
                if last_comma != -1:
                    raw_text = raw_text[:last_comma]
                raw_text += "]}"
            elif raw_text.count("{") > raw_text.count("}"):
                raw_text += "}"
                
            try:
                json.loads(raw_text)
                return raw_text
            except:
                pass

        raise ValueError(f"Failed to parse JSON from extracted text: {text[:100]}...")


    def _local_response(self, system_prompt: str, user_prompt: str, response_format: str, temperature: float) -> str:
        """
        Generates a response using the locally loaded model.
        """
        # Strengthen prompt for JSON generation
        if response_format == "json":
            system_prompt += "\nIMPORTANT: Output ONLY a valid JSON string. Do not include any explanations, preambles, or markdown formatting."
            user_prompt += "\nRespond with raw JSON only."

        # Construct a prompt. This template might need adjustment based on the specific model (e.g. ChatML for Qwen)
        # For simplicity, we'll use a basic structure or the tokenizer's chat template if available.
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Try to use apply_chat_template if the tokenizer supports it
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback for models without chat template in tokenizer config
            prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": 1024, # Limit token generation to avoid infinite loops and truncation
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "return_full_text": False,
            "pad_token_id": self.tokenizer.eos_token_id, # Fix CUDA device-side assert when pad token is missing
            "batch_size": 1, # Avoid multi-pipeline issue on GPU
            "max_length": None # Explicitly override default max_length to avoid conflict with max_new_tokens
        }
        
        outputs = self.pipeline(prompt, **gen_kwargs)
        generated_text = outputs[0]["generated_text"]
        
        # Extract the assistant's response. 
        # Since return_full_text=False, generated_text is just the new content.
        response = generated_text.strip()

        # Try to use the shared JSON cleaner if response_format is JSON
        if response_format == "json":
            response = self._clean_and_extract_json(response)
        
        # Fallback Aggressive JSON truncation recovery if cleaner fails or wasn't used
        # Find the first '{' and the last '}'
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            response = response[start_idx:end_idx+1]
        
        # Sometimes arrays get truncated like [1, 2, 3
        if response.count('[') > response.count(']'):
            response += ']'
        # Sometimes objects get truncated
        if response.count('{') > response.count('}'):
            response += '}'

        return response

    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a fake JSON response based on keywords in the prompt.
        This allows testing the flow without paying for tokens.
        """
        # Detect if this is a Local Agent or Global Agent request
        if "Local Agent" in system_prompt:
            # Simulate a decision to adjust parameters slightly
            return json.dumps({
                "reasoning": "Performance is stable, increasing exploration slightly.",
                "action_type": "adjust_parameters",
                "parameters": {
                    "cr1": 0.4,
                    "cr2": 0.4,
                    "beta": 2.5,
                    "alpha": 10.0
                },
                "candidate_seed_set": None
            })
        
        elif "Global Agent" in system_prompt:
            # Simulate a decision to keep baselines
            return json.dumps({
                "reasoning": "Global convergence is proceeding normally. No merges needed yet.",
                "global_baselines": {
                    "cr1": 0.3, 
                    "cr2": 0.3,
                    "beta": 2.0,
                    "alpha": 12.0
                },
                "budget_adjustments": {},
                "merge_suggestions": []
            })
            
        return "{}"
