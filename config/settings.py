import datetime
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, HttpUrl, SecretStr, field_validator
from typing import Optional
from datetime import datetime, timezone # timezone


class Settings(BaseSettings):
    """.env"""
    

    # ========== Base_Url LLM ============

    # BASE_URL_LLM: str = Field(..., env="BASE_URL") # ADD BASE_URL



    # ========== Elasticsearch ==========
    ELASTIC_HOST: str  = Field("$ELASTIC_HOST", env="${ELASTIC_HOST}")
    ELASTIC_USER: str = Field("ELASTIC_USER", env="${ELASTIC_USER}")
    ELASTIC_PASSWORD: str = Field(..., env="${ELASTIC_PASSWORD}", min_length=1)
    ELASTIC_INDEX_PATTERN: str = Field(
        "${ELASTIC_INDEX_PATTERN}", 
        env="${ELASTIC_INDEX_PATTERN}"
    )
    ELASTIC_API_KEY:str = Field("ELASTIC_API_KEY",env="${ELASTIC_API_KEY}")
    TARGET_INDEX_PREFIX: str = Field("TARGET_INDEX_PREFIX", env="${TARGET_INDEX_PREFIX}")
    
    # ========== Neo4j ==========
    #NEO4J_URI: str = Field("${NEO4J_URI}", env="NEO4J_URI")
    NEO4J_URI: str = Field("NEO4J_URI", env="${NEO4J_URI}")
    NEO4J_USER: str = Field("NEO4J_USER", env="${NEO4J_USER}")
    NEO4J_PASSWORD: str  = Field(..., env="${NEO4J_PASSWORD}", min_length=1)
    
    # ========== Query Parameters ==========
    TIME_RANGE_GTE: str = Field("TIME_RANGE_GTE", env="${TIME_RANGE_GTE}")
    TIME_RANGE_LTE: str = Field("TIME_RANGE_LTE", env="${TIME_RANGE_LTE}")
    MIN_RISK_SCORE: int = Field("MIN_RISK_SCORE", env="${MIN_RISK_SCORE}", ge=0, le=100)
    MAX_CHAINS: int = Field("MAX_CHAINS", env="${MAX_CHAINS}", ge=1, le=500)
    MAX_CHAIN_SIZE: int = Field("${MAX_CHAIN_SIZE}", env="${MAX_CHAIN_SIZE}", ge=1, le=100)


    # ========== Log Level ===============

    LOG_LEVEL: str = Field("LOG_LEVEL", env="${LOG_LEVEL}")
    LOG_FILE: Path = Field(Path("LOG_FILE"), env="${LOG_FILE}")
    
    # ========== SEPSes Ontology Base URI ==========
    SEPSES_BASE_URI: str = Field(
        "https://w3id.org/sepses/vocab/event/log#", 
        env="SEPSES_BASE_URI"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra= "ignore"
        

    # @classmethod
    # @field_validator("ELASTIC_HOST", "NEO4J_URI")
    # def validate_uri_scheme(cls, v , info):
    #     if not v.startswith(("http://", "https://", "bolt://")):
    #         raise ValueError(f"Invalid URI scheme: {v}")
    #     return v


    # --- field_validator Validators ---
    # Added LOG_FILE to validator
    @field_validator("ELASTIC_HOST", "NEO4J_URI", "LOG_FILE") 
    @classmethod
    def validate_uri_and_path(cls, v: str, info) -> str:
        # Check if it's a URI field
        if info.field_name in ("${ELASTIC_HOST}", "${NEO4J_URI}"):
            if not v.startswith(("http://", "https://", "bolt://")):
                raise ValueError(f"Invalid URI scheme for {info.field_name}: {v}")
        # Check if LOG_FILE path is valid (optional, can be more complex)
        elif info.field_name == "LOG_FILE":
            # Ensure the directory exists, create if not
            log_dir = Path(v).parent
            if not log_dir.exists():
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Created log directory: {log_dir}")
                except OSError as e:
                    raise ValueError(f"Could not create log directory {log_dir}: {e}") from e
        return v



    #Generating target index name with current date
    # def get_target_index(self) -> str:
    #     from datetime import datetime
    #     today = datetime.utcnow().strftime("%Y.%m.%d")
    #     return f"{self.TARGET_INDEX_PREFIX}-{today}"
    @staticmethod
    def get_target_index() -> str:
        """Generates the target index name with the current date."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"{settings.TARGET_INDEX_PREFIX}-{today}"

# Singleton instance
settings = Settings()