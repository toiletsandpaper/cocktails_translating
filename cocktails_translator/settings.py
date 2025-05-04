from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    TRANSLATOR_MODEL_NAME: str = Field(default="gemma-3-27b-it-qat", description="Model name for the translator")
    HF_TOKEN: str = Field(description="Hugging Face token")
    
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_API_BASE: HttpUrl = Field(default=HttpUrl("http://127.0.0.1:1234/v1"), description="OpenAI API base URL")
    
    LANGFUSE_SECRET_KEY: str = Field(description="Langfuse secret key")
    LANGFUSE_PUBLIC_KEY: str = Field(description="Langfuse public key")
    LANGFUSE_HOST: HttpUrl = Field(default=HttpUrl("https://api.langfuse.com"), description="Langfuse host")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")
    