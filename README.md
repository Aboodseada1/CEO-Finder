# Standalone Company CEO Finder

A Python script to find the CEO or other key executive of a company using its name or domain. It leverages a SearXNG instance for web searching and various Large Language Model (LLM) providers for analyzing the search results.

## Features

* Finds CEO/President/Founder/Owner names based on public web data.
* Accepts company name or company domain/URL as input.
* Integrates with a SearXNG instance for privacy-respecting web searches.
* Supports multiple LLM providers:
    *   Google Gemini (`gemini`)
    *   OpenAI (`openai`)
    *   Groq (`groq`)
    *   Local Ollama (`ollama`)
* Allows specifying the *exact* LLM model via command-line (`--llm-model`).
* Uses provider-specific fallback model lists if no exact model is specified.
* Standalone script design (dependencies are external libraries).
* Configurable logging level for debugging.
* Outputs results in JSON format.

## Prerequisites

1. **Python:** Python 3.7+ is recommended.
2. **Pip:** Python package installer.
3. **SearXNG Instance:** You need access to a running SearXNG instance. This could be self-hosted or a public instance (use public instances responsibly). Note the base URL of the instance (e.g., `http://127.0.0.1:8080`).
4. **LLM API Keys (Conditional):**
    *   If using `gemini`, `openai`, or `groq`, you need valid API keys for those services.
5. **Ollama (Conditional):**
    *   If using the `ollama` provider, you need Ollama installed, running, and the desired model(s) pulled (e.g., `ollama run llama3:8b`). See [Ollama GitHub](https://github.com/ollama/ollama).

## Installation

1. **Clone the repository or download the script:**
2. ```bash
   git clone <repository_url> # Replace with your actual repo URL
   cd <repository_directory>
   ```
3. Or simply download the `standalone_ceo_finder.py` file.
4. **Install required Python packages:**
5. * **Core dependency:**
   * ```bash
     pip install requests
     ```
   * **Install based on the LLM provider(s) you plan to use:**
   * ```bash
     # For Gemini
     pip install google-generativeai
     
     # For OpenAI or Groq (Groq uses OpenAI's client library structure)
     pip install openai
     ```
   * **Or install all potential LLM libraries:**
   * ```bash
     pip install requests google-generativeai openai
     ```
   * *(Optional but recommended)* Create and use a Python virtual environment:
   * ```bash
     python -m venv venv
     source venv/bin/activate # On Windows use `venv\Scripts\activate`
     pip install -r requirements.txt # If a requirements.txt file is provided
     # Or install manually as shown above
     ```

## Usage

The script is run from the command line.

```bash
python standalone_ceo_finder.py <company_input> --searx-url <searx_instance_url> --llm-provider <provider_name> [options]
```

**Arguments:**

* `company_input`: (Required) The company name (e.g., `"Apple Inc."` ) or domain/URL (e.g., `google.com`). Wrap names with spaces in quotes.
* `--searx-url`: (Required) The base URL of your running SearXNG instance (e.g., `http://127.0.0.1:8080`, `https://searx.example.org`).
* `--llm-provider`: (Required) The LLM provider to use. Choices: `gemini`, `openai`, `groq`, `ollama`.
* `--llm-model`: (Optional) Specify the exact LLM model name to use (e.g., `gpt-4o-mini`, `gemini-1.5-flash`, `llama3-70b-8192`, `phi3:latest`). If omitted, the script uses the provider's internal fallback list.
* `--llm-api-key`: (Optional) The API key for the chosen LLM provider. **Required** if `--llm-provider` is `gemini`, `openai`, or `groq`. Not used for `ollama`.
* `--log-level`: (Optional) Set the logging level. Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default is `INFO`.

## Examples

*(Replace placeholders like `http://localhost:8080`, `YOUR_API_KEY`, and company names)*

### Using OpenAI with specific model:

```bash
python standalone_ceo_finder.py "Nvidia" \
    --searx-url "http://localhost:8080" \
    --llm-provider openai \
    --llm-model "gpt-4o-mini" \
    --llm-api-key "YOUR_OPENAI_API_KEY"
```

### Using OpenAI with provider fallback:

```bash
python standalone_ceo_finder.py "Microsoft" \
    --searx-url "http://localhost:8080" \
    --llm-provider openai \
    --llm-api-key "YOUR_OPENAI_API_KEY"
```

### Using Gemini with specific model:

```bash
python standalone_ceo_finder.py "Alphabet Inc." \
    --searx-url "https://searx.example.org" \
    --llm-provider gemini \
    --llm-model "gemini-1.5-flash" \
    --llm-api-key "YOUR_GEMINI_API_KEY"
```

### Using Groq with specific model:

```bash
python standalone_ceo_finder.py "Meta Platforms" \
    --searx-url "http://localhost:8080" \
    --llm-provider groq \
    --llm-model "llama3-8b-8192" \
    --llm-api-key "YOUR_GROQ_API_KEY"
```

### Using Ollama with specific local model (ensure 'phi3' is pulled):

```bash
python standalone_ceo_finder.py "Hugging Face" \
    --searx-url "http://localhost:8080" \
    --llm-provider ollama \
    --llm-model "phi3"
```

### Using Ollama with default model (llama3:8b):

```bash
python standalone_ceo_finder.py "GitHub" \
    --searx-url "http://localhost:8080" \
    --llm-provider ollama
```

### Using a domain as input and debug logging:

```bash
python standalone_ceo_finder.py "openai.com" \
    --searx-url "http://localhost:8080" \
    --llm-provider openai \
    --llm-api-key "YOUR_OPENAI_API_KEY" \
    --log-level DEBUG
```

## How It Works

1. **Input Processing:** The script takes the company name or domain. If a domain/URL is detected, it attempts to sanitize it and extract a potential company name.
2. **Query Generation:** It generates several search queries tailored to finding CEO/executive information (e.g., `"Who is the CEO of <Company>"`).
3. **SearXNG Search:** For each query, it sends a request to the specified SearXNG instance to fetch web search results.
4. **LLM Prompting:** It compiles the titles, URLs, and snippets from the search results into a detailed prompt for the LLM, asking it to analyze the information and identify the most likely CEO/executive, prioritizing roles (CEO > President > Founder > Owner) and consistency.
5. **LLM Interaction:** It sends the prompt to the specified LLM provider, using either the specific model requested via `--llm-model` or iterating through the provider's fallback list until a successful response is received.
6. **Response Parsing:** It attempts to parse the LLM's response, specifically looking for a JSON object containing the `ceo_name` key (which can be the name or `null`).
7. **Output:** It prints the final result as a JSON string containing the found `ceo_name` (or `null` if not found) and the `source_model` that provided the answer.

## Configuration (Internal)

While the primary configuration is done via command-line arguments, the script contains internal fallback lists for models if the `--llm-model` argument is *not* used:

* `GEMINI_MODEL_FALLBACK`
* `OPENAI_MODEL_FALLBACK`
* `GROQ_MODEL_FALLBACK`
* `OLLAMA_DEFAULT_MODEL`

You can modify these lists directly within the `standalone_ceo_finder.py` script if you want to change the default models or their fallback order for different providers when not specifying a model explicitly.

## Dependencies

* `requests`
* `google-generativeai` (Optional, needed for `gemini` provider)
* `openai` (Optional, needed for `openai` and `groq` providers)

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request on [GitHub](https://github.com/Aboodseada1?tab=repositories).

## Support Me

If you find this tool useful, consider supporting its development via [PayPal](http://paypal.me/aboodseada1999). Thank you!

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) [Year] abd-el-rahman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```