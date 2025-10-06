import sys
import google.generativeai as genai

MODEL_NAME = "gemini-2.5-flash-lite"

def test_google_api_key(api_key):
    """
    Tests if the provided Google API key works for LLM calls to gemini-2.5-flash-lite.

    Args:
        api_key (str): Google API key to test

    Returns:
        tuple: (bool, str) indicating success status and message
    """
    try:
        genai.configure(api_key=api_key)

        # Initialize the model
        model = genai.GenerativeModel(MODEL_NAME)

        # Minimal test prompt
        response = model.generate_content(
            "Reply with exactly: API_KEY_OK",
            request_options={"timeout": 10},
        )

        # Validate response content
        text = getattr(response, "text", None)
        print(response)
        if text and "API_KEY_OK" in text:
            return True, f"API key is valid. Model '{MODEL_NAME}' responded successfully."
        elif text:
            # We got a response, but not the expected content
            return True, f"API key is valid, but unexpected response from '{MODEL_NAME}': {text!r}"
        else:
            return False, "API key validation failed: Empty or safety-blocked response."

    except Exception as e:
        # Normalize error text for pattern matching
        err = str(e).lower()
        print(e)

        if "401" in err or "unauthorized" in err or "invalid api key" in err:
            return False, "API key validation failed: Invalid API key (401 Unauthorized)."
        if "403" in err or "permission" in err or "not enabled" in err:
            return False, ("API key valid but permission denied (403). "
                           "Ensure the Generative Language API is enabled for your project "
                           "and the key has access.")
        if "429" in err or "quota" in err or "rate limit" in err:
            return False, "API key validation failed: Quota exceeded or rate-limited (429)."
        if "404" in err or "not found" in err or "model" in err and "not found" in err:
            return False, (f"Model '{MODEL_NAME}' not found or not available to your project (404). "
                           "Verify the model name and availability in your account/region.")
        return False, f"API key validation failed: {e}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_google_api_key.py <API_KEY>")
        sys.exit(1)

    api_key = sys.argv[1]
    is_valid, message = test_google_api_key(api_key)
    print(f"Validation Result: {message}")
    sys.exit(0 if is_valid else 1)