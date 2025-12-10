from dotenv import load_dotenv
import os

load_dotenv(override=True)

PROPLANET_API_URL = os.getenv("PROPLANET_API_URL")