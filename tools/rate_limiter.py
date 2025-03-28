import time
import random
import asyncio
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from config import MAX_CALLS_PER_MINUTE, MAX_RETRIES, INITIAL_RETRY_DELAY

class RateLimiter:
    def __init__(self, max_calls_per_minute=MAX_CALLS_PER_MINUTE, max_retries=MAX_RETRIES, initial_delay=INITIAL_RETRY_DELAY):
        self.max_calls_per_minute = max_calls_per_minute
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.calls = []
    
    async def wait_if_needed(self):
        """Wait if we've made too many calls in the last minute"""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]
        
        if len(self.calls) >= self.max_calls_per_minute:
            wait_time = 60 - (now - self.calls[0]) + 1  # Add 1 second buffer
            print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
        
        # Add this call
        self.calls.append(time.time())
    
    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                await self.wait_if_needed()
                return await func(*args, **kwargs)
            except (ResourceExhausted, ServiceUnavailable) as e:
                delay = self.initial_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"API quota exceeded. Retrying in {delay:.2f} seconds... (Attempt {attempt+1}/{self.max_retries})")
                await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        raise Exception(f"Failed after {self.max_retries} attempts due to API quota limits")

# Create a global rate limiter instance
rate_limiter = RateLimiter()
