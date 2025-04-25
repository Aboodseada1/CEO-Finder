# Standalone module to find Company CEO name using Company Name or Domain
# Integrates SearXNG search, LLM analysis, and URL validation.
# Allows specifying a specific LLM model via command line.

import json
import time
import random
import logging
import re
import os
import sys
import requests
import html
import traceback
import argparse
from urllib.parse import urlparse, quote_plus

# --- Configuration ---
# LLM Model Fallbacks (Used if --llm-model is NOT specified)
GEMINI_MODEL_FALLBACK = ["gemini-1.5-flash", "gemini-pro"]
OPENAI_MODEL_FALLBACK = ["gpt-4o-mini", "gpt-3.5-turbo"]
GROQ_MODEL_FALLBACK = ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
OLLAMA_DEFAULT_MODEL = "llama3:8b" # Default if provider is ollama and no specific model

# Retry Configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 10

# Max prompt length
MAX_PROMPT_LENGTH = 30000

# --- Optional Imports ---
try:
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    import openai
except ImportError:
    openai = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Integrated URL Sanitization Logic ---
def sanitize_url(url):
    if not isinstance(url, str) or not url.strip():
        logger.warning("Invalid URL input: empty or not a string.")
        return None
    if not url.startswith('http://') and not url.startswith('https://'):
        if url.startswith('//'):
            url = 'https:' + url
        else:
            url = 'https://' + url
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        if domain.lower().startswith('www.'):
            domain = domain[4:]
        domain = re.sub(r':\d+$', '', domain)
        if not domain or '.' not in domain or len(domain.split('.')[-1]) < 2:
            logger.warning(f"URL '{url}' resulted in invalid domain '{domain}'")
            return None
        return domain.lower()
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return None

# --- Integrated SearXNG Client Logic ---
class SearXNGClient:
    def __init__(self, base_url):
        if not base_url or not base_url.startswith(('http://', 'https://')):
             raise ValueError("Invalid SearXNG base URL provided.")
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'application/json'
        })
        logger.info(f"SearXNGClient initialized with base URL: {self.base_url}")

    def search(self, query, max_pages=5, timeout=10):
        all_results = []
        page = 1
        last_error = None
        while page <= max_pages:
            encoded_query = quote_plus(query)
            url = f"{self.base_url}/search?q={encoded_query}&format=json&pageno={page}"
            try:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    for result in data['results']:
                        simplified_result = {
                            'title': result.get('title', ''),
                            'url': result.get('url', ''),
                            'content': result.get('content', ''),
                            'score': result.get('score', 0)
                        }
                        all_results.append(simplified_result)
                    page += 1
                    time.sleep(0.2)
                else:
                    break
            except requests.exceptions.RequestException as e:
                last_error = f"SearXNG request failed page {page}: {e}"
                logger.error(last_error)
                break
            except json.JSONDecodeError as e:
                last_error = f"SearXNG JSON decode error page {page}: {e}"
                logger.error(last_error)
                break
            except Exception as e:
                last_error = f"Unexpected SearXNG error page {page}: {e}"
                logger.error(last_error)
                break
        result_data = {"query": query, "num_results": len(all_results), "results": all_results}
        if last_error and not all_results:
            result_data["error"] = last_error
        return result_data

def search_web_standalone(query, searx_base_url, pages=2, max_retries=2):
    if not searx_base_url:
        logger.error("No SearXNG base URL provided.")
        return json.dumps({"query": query, "num_results": 0, "results": [], "error": "Missing SearXNG URL"})
    try:
        client = SearXNGClient(searx_base_url)
    except ValueError as e:
        logger.error(f"Failed to initialize SearXNGClient: {e}")
        return json.dumps({"query": query, "num_results": 0, "results": [], "error": str(e)})
    attempt = 0
    results = {}
    while attempt < max_retries:
        results = client.search(query, max_pages=pages)
        if results["num_results"] > 0:
            logger.info(f"SearXNG search successful for '{query}' on attempt {attempt+1}")
            return json.dumps(results)
        elif "error" in results:
             logger.warning(f"SearXNG search error for '{query}' on attempt {attempt+1}: {results['error']}")
        else:
            logger.warning(f"SearXNG search for '{query}' yielded 0 results on attempt {attempt+1}")
        attempt += 1
        if attempt < max_retries:
            delay = INITIAL_RETRY_DELAY * (2 ** (attempt - 1))
            logger.info(f"Retrying SearXNG search after {delay:.2f}s...")
            time.sleep(delay)
    final_error = results.get("error", "No search results found after multiple attempts.")
    no_results_data = {"query": query, "num_results": 0, "results": [], "error": final_error}
    logger.error(f"SearXNG search failed permanently for '{query}': {final_error}")
    return json.dumps(no_results_data)

# --- Integrated LLM Call Logic ---
def sanitize_prompt(prompt):
    if not isinstance(prompt, str): prompt = str(prompt)
    prompt = prompt[:MAX_PROMPT_LENGTH]
    prompt = prompt.replace('\0', '')
    prompt = html.escape(prompt)
    prompt = ''.join(c for c in prompt if ord(c) >= 32 or c in '\n\r\t')
    prompt = re.sub(r'\s+', ' ', prompt).strip()
    return prompt

def _call_llm(prompt, provider, api_key=None, model_name=None): # Added model_name
    """Routes the LLM call, potentially using a specific model."""
    sanitized_prompt = sanitize_prompt(prompt)
    if not sanitized_prompt:
        return None, "Invalid or empty prompt after sanitization"

    provider = provider.lower()
    if model_name:
        logger.info(f"Attempting LLM call via provider: {provider}, specific model: {model_name}")
    else:
        logger.info(f"Attempting LLM call via provider: {provider} (using fallback)")

    response = None
    model_used_info = f"{provider}:unknown"
    error_message = f"Provider '{provider}' failed"

    try:
        if provider == "gemini":
            if not genai: raise ImportError("google.generativeai required.")
            if not api_key: raise ValueError("API key required for Gemini.")
            response, model_used_info = _get_gemini_response(sanitized_prompt, api_key, model_name)
        elif provider == "openai":
            if not openai: raise ImportError("openai required.")
            if not api_key: raise ValueError("API key required for OpenAI.")
            response, model_used_info = _get_openai_response(sanitized_prompt, api_key, model_name)
        elif provider == "groq":
            if not api_key: raise ValueError("API key required for Groq.")
            response, model_used_info = _get_groq_response(sanitized_prompt, api_key, model_name)
        elif provider == "ollama":
            response, model_used_info = _get_ollama_response(sanitized_prompt, model_name)
        else:
            error_message = f"Unsupported LLM provider: {provider}"
            logger.error(error_message)
            return None, error_message

        if response:
            return response, model_used_info
        else:
            return None, f"{provider.capitalize()} request failed for model(s) {model_name or 'fallback'}"

    except ImportError as e:
        error_message = f"Missing library for {provider}: {e}"
        logger.error(error_message)
        return None, error_message
    except ValueError as e:
        error_message = f"Configuration error for {provider}: {e}"
        logger.error(error_message)
        return None, error_message
    except Exception as e:
        error_message = f"Unexpected error during {provider} call: {e}"
        logger.exception(error_message)
        return None, error_message

# --- Provider-Specific LLM Functions ---
def _get_gemini_response(prompt, api_key, model_name=None):
    """Gets response from Gemini. Uses specific model if provided, else fallback."""
    genai.configure(api_key=api_key)
    last_exception = None
    models_to_try = [model_name] if model_name else GEMINI_MODEL_FALLBACK

    for current_model in models_to_try:
        logger.info(f"Trying Gemini model: {current_model}")
        try:
             model = genai.GenerativeModel(current_model)
        except Exception as init_e:
             logger.error(f"Failed to initialize Gemini model '{current_model}': {init_e}")
             last_exception = init_e
             continue # Try next model if using fallback

        for attempt in range(MAX_RETRIES):
            try:
                response = model.generate_content(prompt)
                logger.info(f"Gemini ({current_model}) generation successful.")
                return json.dumps({"response": response.text}), f"gemini:{current_model}"
            except Exception as e:
                last_exception = e
                logger.warning(f"Gemini ({current_model}) attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    delay = min(INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 0.5), MAX_RETRY_DELAY)
                    time.sleep(delay)
                else:
                    break # Failed all retries for this specific model
        # If loop finishes for this model, and we were given a specific model, stop.
        if model_name:
             break

    logger.error(f"Gemini failed for model(s): {models_to_try}. Last error: {last_exception}")
    return None, f"gemini:failed ({last_exception})"

def _get_openai_response(prompt, api_key, model_name=None):
    """Gets response from OpenAI. Uses specific model if provided, else fallback."""
    client = openai.OpenAI(api_key=api_key)
    last_exception = None
    models_to_try = [model_name] if model_name else OPENAI_MODEL_FALLBACK

    for current_model in models_to_try:
        logger.info(f"Trying OpenAI model: {current_model}")
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                logger.info(f"OpenAI ({current_model}) generation successful.")
                response_text = response.choices[0].message.content
                return json.dumps({"response": response_text}), f"openai:{current_model}"
            except Exception as e:
                last_exception = e
                logger.warning(f"OpenAI ({current_model}) attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    delay = min(INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 0.5), MAX_RETRY_DELAY)
                    time.sleep(delay)
                else:
                    break # Failed all retries for this model
        if model_name:
            break

    logger.error(f"OpenAI failed for model(s): {models_to_try}. Last error: {last_exception}")
    return None, f"openai:failed ({last_exception})"

def _get_groq_response(prompt, api_key, model_name=None):
    """Gets response from Groq. Uses specific model if provided, else fallback."""
    client = openai.OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    last_exception = None
    models_to_try = [model_name] if model_name else GROQ_MODEL_FALLBACK

    for current_model in models_to_try:
        logger.info(f"Trying Groq model: {current_model}")
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                logger.info(f"Groq ({current_model}) generation successful.")
                response_text = response.choices[0].message.content
                return json.dumps({"response": response_text}), f"groq:{current_model}"
            except Exception as e:
                last_exception = e
                logger.warning(f"Groq ({current_model}) attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
                if "rate limit" in str(e).lower() or "429" in str(e):
                    logger.error(f"Groq rate limit hit for key ...{api_key[-4:]}, model {current_model}. Stopping.")
                    # If specific model was requested, return failure now
                    if model_name:
                        return None, f"groq:{current_model} rate_limited"
                    break # Stop retries for this model, try next in fallback if possible
                if attempt < MAX_RETRIES - 1:
                    delay = min(INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 0.5), MAX_RETRY_DELAY)
                    time.sleep(delay)
                else:
                    break # Failed all retries for this model
        if model_name:
             break

    logger.error(f"Groq failed for model(s): {models_to_try}. Last error: {last_exception}")
    return None, f"groq:failed ({last_exception})"

def _get_ollama_response(prompt, model_name=None):
    """Gets response from local Ollama model. Uses specific model if provided, else default."""
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
    # Use specified model, or the default, or fallback if default is None
    ollama_model_to_use = model_name if model_name else OLLAMA_DEFAULT_MODEL
    if not ollama_model_to_use:
         logger.error("No Ollama model specified or defaulted.")
         return None, "ollama:no_model_specified"

    logger.info(f"Trying Ollama model: {ollama_model_to_use} at {ollama_url}")
    try:
        response = requests.post(
            ollama_url,
            json={"model": ollama_model_to_use, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        response_json = response.json()
        logger.info(f"Ollama ({ollama_model_to_use}) generation successful.")
        response_text = response_json.get('response', '').strip()
        return json.dumps({"response": response_text}), f"ollama:{ollama_model_to_use}"
    except requests.exceptions.ConnectionError:
        logger.error(f"Ollama connection failed at {ollama_url}. Is Ollama running?")
        return None, f"ollama:{ollama_model_to_use}_connection_error"
    except Exception as e:
        logger.error(f"Ollama ({ollama_model_to_use}) failed: {e}")
        # Check if model not found error (common issue)
        if response and response.status_code == 404:
             logger.error(f"Model '{ollama_model_to_use}' likely not found by Ollama server.")
             return None, f"ollama:{ollama_model_to_use}_not_found"
        return None, f"ollama:{ollama_model_to_use}_failed ({e})"

# --- Core CEO Finder Logic ---
class CEOFinder:
    def __init__(self, searx_base_url, llm_provider, llm_api_key=None, llm_model_name=None, delay_range=(1, 3), max_search_pages=2): # Added llm_model_name
        if not searx_base_url: raise ValueError("SearXNG base URL required.")
        if not llm_provider: raise ValueError("LLM provider required.")

        self.searx_base_url = searx_base_url
        self.llm_provider = llm_provider.lower()
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name # Store the specific model name
        self.delay_range = delay_range
        self.max_search_pages = max_search_pages
        log_model = f", Model: {self.llm_model_name}" if self.llm_model_name else " (using fallback)"
        logger.info(f"CEOFinder initialized with SearXNG URL: {searx_base_url}, LLM Provider: {self.llm_provider}{log_model}")

    def generate_search_queries(self, company_name):
        return [
            f'"{company_name}" CEO official name',
            f'Who is the CEO of "{company_name}" company',
            f'"{company_name}" executive leadership team',
            f'"{company_name}" founder OR president OR owner',
            f'site:linkedin.com/in "CEO" "{company_name}"',
            f'{company_name} CEO',
            f'Owner of {company_name}',
        ]

    def create_analysis_prompt(self, company_name, search_results):
        prompt = (
            f"Analyze the following search result snippets to determine the current CEO (Chief Executive Officer), President, Owner, or Founder of the company named or closely related to \"{company_name}\".\n\n"
            "Search Results:\n"
            "---------------\n"
        )
        max_results_in_prompt = 15
        results_to_include = search_results[:max_results_in_prompt]
        if not results_to_include:
             prompt += "No search results provided.\n\n"
        else:
            for i, result in enumerate(results_to_include):
                prompt += f"Result {i+1}:\n"
                prompt += f"  Title: {result.get('title', 'N/A')}\n"
                prompt += f"  URL: {result.get('url', 'N/A')}\n"
                prompt += f"  Snippet: {result.get('content', 'N/A')}\n\n"
        prompt += (
            "Instructions:\n"
            "1. Identify potential names associated with high-level executive roles (CEO, President, Owner, Founder) for the target company \"{company_name}\".\n"
            "2. Prioritize the role: CEO > President > Founder > Owner.\n"
            "3. Look for consistency. Choose the name mentioned most frequently or in authoritative sources.\n"
            "4. Verify the company context. Ensure the person is linked to \"{company_name}\".\n"
            "5. Extract the **full name** (First and Last Name).\n"
            "6. If no likely candidate is found, return null.\n"
            "7. Exclude generic titles or placeholder names.\n"
            "8. Format the output STRICTLY as a JSON object: `{{\"ceo_name\": \"Full Name\"}}` or `{{\"ceo_name\": null}}`.\n"
            "9. Do NOT include explanations outside the JSON object.\n\n"
            "JSON Output:"
        )
        return prompt

    def clean_llm_json_response(self, llm_response_text):
        if not llm_response_text or not isinstance(llm_response_text, str): return None
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_response_text, re.DOTALL | re.IGNORECASE)
        if match:
            json_text = match.group(1)
        else:
            match = re.search(r'\{.*?\}', llm_response_text, re.DOTALL)
            if match: json_text = match.group(0)
            else:
                logger.warning(f"Could not find JSON object in LLM response: {llm_response_text[:200]}...")
                return None
        try:
            parsed_json = json.loads(json_text)
            if isinstance(parsed_json, dict) and "ceo_name" in parsed_json:
                 if parsed_json["ceo_name"] and isinstance(parsed_json["ceo_name"], str):
                     name = parsed_json["ceo_name"].strip().title()
                     if name.lower() in ["john doe", "jane doe", "john smith", "null", "none", "n/a", ""]:
                         logger.warning(f"LLM returned placeholder/empty name: {name}")
                         parsed_json["ceo_name"] = None
                     else:
                         parsed_json["ceo_name"] = name
                 elif parsed_json["ceo_name"] is not None:
                      logger.warning(f"LLM returned non-string/null ceo_name: {parsed_json['ceo_name']}")
                      parsed_json["ceo_name"] = None
                 return parsed_json
            else:
                 logger.warning(f"Extracted JSON invalid or missing 'ceo_name': {json_text}")
                 return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode extracted JSON: {e}. Text: {json_text}")
            return None
        except Exception as e:
             logger.error(f"Unexpected error cleaning LLM response: {e}")
             return None

    def find_ceo(self, company_input):
        if not company_input or not isinstance(company_input, str):
            logger.warning("Empty or invalid company input.")
            return {"ceo_name": None, "source_model": None}
        company_input = company_input.strip()
        company_name = company_input
        domain = None
        if '.' in company_input and ('/' in company_input or any(company_input.endswith(tld) for tld in ['.com', '.org', '.net', '.io', '.co', '.ai'])):
            sanitized_domain = sanitize_url(company_input)
            if sanitized_domain:
                company_name = sanitized_domain.split('.')[0]
                domain = sanitized_domain
                logger.info(f"Input is domain. Using name '{company_name}', domain '{domain}'.")
            else:
                 logger.warning(f"Input looked like domain but failed sanitization: '{company_input}'. Treating as name.")
                 company_name = company_input
        else:
             company_name = company_input
        logger.info(f"Searching for CEO related to: '{company_name}'" + (f" (Domain: {domain})" if domain else ""))
        queries = self.generate_search_queries(company_name)
        source_model_used = None
        for i, query in enumerate(queries):
            logger.info(f"--- Running Query {i+1}/{len(queries)}: {query} ---")
            search_results_json = search_web_standalone(query, self.searx_base_url, pages=self.max_search_pages)
            try:
                search_data = json.loads(search_results_json)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode search results JSON for query: {query}")
                search_data = {"results": [], "error": "JSON decode error"}
            results = search_data.get("results", [])
            if not results:
                log_func = logger.warning if not search_data.get("error") else logger.error
                log_func(f"No results or error for query: {query}. Error: {search_data.get('error', 'N/A')}")
                if i < len(queries) - 1:
                     delay = random.uniform(*self.delay_range)
                     time.sleep(delay)
                continue
            analysis_prompt = self.create_analysis_prompt(company_name, results)
            # Pass the specific model name here
            llm_response_json, model_used = _call_llm(analysis_prompt, self.llm_provider, self.llm_api_key, self.llm_model_name)
            source_model_used = model_used # Store model used for this attempt
            if not llm_response_json:
                logger.warning(f"LLM ({model_used or self.llm_provider}) failed analysis for query: {query}")
                if i < len(queries) - 1:
                     delay = random.uniform(*self.delay_range)
                     time.sleep(delay)
                continue
            try:
                llm_data = json.loads(llm_response_json)
                raw_llm_output = llm_data.get("response")
                if not raw_llm_output:
                     logger.warning(f"LLM response content empty for query: {query}")
                     continue
                ceo_data = self.clean_llm_json_response(raw_llm_output)
                if ceo_data and ceo_data.get("ceo_name"):
                    found_name = ceo_data["ceo_name"]
                    logger.info(f"Successfully found CEO name: '{found_name}' for '{company_name}' using query '{query}' and model '{model_used}'")
                    return {"ceo_name": found_name, "source_model": model_used}
                elif ceo_data and ceo_data.get("ceo_name") is None:
                     logger.info(f"LLM explicitly returned null for CEO for query '{query}'. Trying next.")
                else:
                    logger.warning(f"Could not extract valid CEO JSON from LLM response for query '{query}'. Response: {raw_llm_output[:200]}...")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode outer JSON wrapper from LLM for query: {query}")
            except Exception as e:
                logger.error(f"Unexpected error processing LLM response for query {query}: {e}")
            if i < len(queries) - 1:
                delay = random.uniform(*self.delay_range)
                logger.info(f"Waiting {delay:.2f}s before next query.")
                time.sleep(delay)
        logger.warning(f"No definitive CEO name found for '{company_name}' after trying all queries.")
        return {"ceo_name": None, "source_model": source_model_used}

# --- Public Function Interfaces ---
def find_company_ceo(company_input, searx_url, llm_provider, llm_api_key=None, llm_model_name=None): # Added llm_model_name
    """High-level function to find CEO, optionally using a specific model."""
    try:
        finder = CEOFinder(
            searx_base_url=searx_url,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name # Pass model name here
        )
        result = finder.find_ceo(company_input)
        return json.dumps(result)
    except ValueError as e:
        logger.error(f"Initialization error: {e}")
        return json.dumps({"ceo_name": None, "source_model": None, "error": str(e)})
    except Exception as e:
        logger.error(f"Unhandled exception in find_company_ceo: {e}")
        logger.error(traceback.format_exc())
        return json.dumps({"ceo_name": None, "source_model": None, "error": "An unexpected error occurred"})

def extract_ceo_name(ceo_json_string):
    """Extracts CEO name from result JSON string."""
    if not ceo_json_string: return None
    try:
        ceo_data = json.loads(ceo_json_string)
        return ceo_data.get("ceo_name")
    except json.JSONDecodeError:
        logger.error(f"Failed to parse CEO JSON string: {ceo_json_string}")
        return None
    except Exception as e:
        logger.error(f"Error extracting CEO name: {e}")
        return None

# --- Example Command-Line Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the CEO of a company using SearXNG and an LLM.")
    parser.add_argument("company_input", help="The company name or domain.")
    parser.add_argument("--searx-url", required=True, help="Base URL of the SearXNG instance (e.g., http://localhost:8080)")
    parser.add_argument("--llm-provider", required=True, choices=['gemini', 'openai', 'groq', 'ollama'], help="LLM provider to use.")
    parser.add_argument("--llm-model", help="Specific LLM model name (e.g., 'gpt-4o-mini', 'gemini-1.5-flash'). If omitted, provider's fallback list is used.")
    parser.add_argument("--llm-api-key", help="API key for the selected LLM provider (required for gemini, openai, groq).")
    parser.add_argument("--log-level", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help="Set logging level.")

    args = parser.parse_args()

    log_level_map = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
    logger.setLevel(log_level_map[args.log_level])
    logging.getLogger().setLevel(log_level_map[args.log_level]) # Set root logger level too

    if args.llm_provider in ['gemini', 'openai', 'groq'] and not args.llm_api_key:
        print(f"Error: API key (--llm-api-key) required for provider '{args.llm_provider}'.", file=sys.stderr)
        sys.exit(1)
    if args.llm_provider == 'gemini' and not genai:
         print("Error: 'google-generativeai' package required for Gemini. Install: pip install google-generativeai", file=sys.stderr)
         sys.exit(1)
    if (args.llm_provider == 'openai' or args.llm_provider == 'groq') and not openai:
         print(f"Error: 'openai' package required for {args.llm_provider.capitalize()}. Install: pip install openai", file=sys.stderr)
         sys.exit(1)

    print(f"Searching for CEO of: '{args.company_input}'")
    print(f"Using SearXNG at: {args.searx_url}")
    print(f"Using LLM Provider: {args.llm_provider}")
    if args.llm_model:
        print(f"Using Specific LLM Model: {args.llm_model}")
    else:
        print("Using LLM Provider's fallback model list.")

    start_time = time.time()
    ceo_json_result = find_company_ceo(
        company_input=args.company_input,
        searx_url=args.searx_url,
        llm_provider=args.llm_provider,
        llm_api_key=args.llm_api_key,
        llm_model_name=args.llm_model # Pass the specific model name here
    )
    end_time = time.time()

    print("\n--- Result ---")
    try:
        parsed_result = json.loads(ceo_json_result)
        print(json.dumps(parsed_result, indent=2))
        ceo_name = parsed_result.get("ceo_name")
        model = parsed_result.get("source_model", "N/A")
        print(f"\nExtracted CEO Name: {ceo_name if ceo_name else 'Not Found'}")
        print(f"Source Model Used: {model}")
    except json.JSONDecodeError:
        print("Error: Failed to parse final JSON result.")
        print(f"Raw output: {ceo_json_result}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")