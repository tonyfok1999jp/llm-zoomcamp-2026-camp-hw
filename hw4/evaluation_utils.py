import threading
import time

from google.genai import types as genai_types
from google.genai import errors as genai_errors
from tqdm.auto import tqdm
from rag_helper import RAGBase


class RateLimiter:
    """Thread-safe limiter that spaces out calls to at most max_per_minute."""

    def __init__(self, max_per_minute):
        self.interval = 60.0 / max_per_minute
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_time)
            self.next_time = start + self.interval
            sleep_for = start - now

        if sleep_for > 0:
            time.sleep(sleep_for)


_gemini_rate_limiter = RateLimiter(max_per_minute=15)


def calc_price(usage):
    input_price_per_million = 0.75
    output_price_per_million = 4.50

    input_cost = (usage.input_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


def calc_price_gemini(usage):
    # gemini-3.1-flash-lite pricing; check current rates at ai.google.dev/pricing.
    input_price_per_million = 0.10
    output_price_per_million = 0.40

    output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)

    input_cost = (usage.prompt_token_count / 1_000_000) * input_price_per_million
    output_cost = (output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price_gemini(usages):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price_gemini(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


def llm_structured_gemini(client, instructions, user_prompt, output_type, model="gemini-3.1-flash-lite"):
    _gemini_rate_limiter.wait()

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=instructions,
            response_mime_type="application/json",
            response_schema=output_type,
        ),
    )

    return response.parsed, response.usage_metadata


def llm_structured_gemini_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="gemini-3.1-flash-lite",
    max_retries=5,
):
    for attempt in range(max_retries):
        try:
            return llm_structured_gemini(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except genai_errors.ClientError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(15 * (attempt + 1) if e.code == 429 else 2 ** attempt)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def llm(self, prompt):
        _gemini_rate_limiter.wait()

        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=self.instructions,
            ),
        )

        self.last_usage = response.usage_metadata
        self.usages.append(response.usage_metadata)

        return response.text

    def total_cost(self):
        return calc_total_price_gemini(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results