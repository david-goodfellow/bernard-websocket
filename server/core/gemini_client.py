# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gemini client initialization and connection management
"""

import logging
import os

from config.config import CONFIG, MODEL, ConfigurationError, api_config
from google import genai
from core.mcp_session import create_mcp_session

logger = logging.getLogger(__name__)


async def create_gemini_session(shared_mcp_session=None):
    """Create and initialize the Gemini client and session"""
    try:
        # Initialize authentication
        await api_config.initialize()

        if api_config.use_vertex:
            # Vertex AI configuration
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")

            if not project_id:
                raise ConfigurationError(
                    "GOOGLE_CLOUD_PROJECT is required for Vertex AI"
                )

            logger.info(
                f"Initializing Vertex AI client with location: {location}, project: {project_id}"
            )

            # Initialize Vertex AI client
            client = genai.Client(
                vertexai=True,
                location=location,
                project=project_id,
            )
        else:
            # API Key configuration - use the key from api_config instead of environment
            if not api_config.api_key:
                raise ConfigurationError("No API key available from Secret Manager or environment")

            logger.info("Initializing Gemini client with API key")
            client = genai.Client(api_key=api_config.api_key)

        # Use shared MCP session if provided, otherwise create new one
        mcp_session = shared_mcp_session
        if not mcp_session:
            # Temporarily disable MCP session to test if it's causing the issue
            # mcp_session = await create_mcp_session()
            mcp_session = None

        # Prepare tools configuration
        tools_config = CONFIG.get("tools", []).copy()
        
        # Add MCP tool if session is available
        if mcp_session:
            from core.mcp_session import create_mcp_tool
            mcp_tool = create_mcp_tool(mcp_session)
            
            # Find the tool that has function_declarations
            function_tool = None
            for tool in tools_config:
                if "function_declarations" in tool:
                    function_tool = tool
                    break
            
            # Merge MCP function declarations into the existing function_declarations tool
            if function_tool:
                function_tool["function_declarations"].extend(mcp_tool["function_declarations"])
                logger.info("MCP function declarations merged into existing tool")
                logger.info(f"Total function declarations: {len(function_tool['function_declarations'])}")
            else:
                # If no existing function_declarations tool, add MCP tool as new tool
                tools_config.append(mcp_tool)
                logger.info("MCP tool added as new tool")
                logger.info(f"Total function declarations: {len(mcp_tool['function_declarations'])}")
        else:
            # Find the tool that has function_declarations for logging
            function_tool = None
            for tool in tools_config:
                if "function_declarations" in tool:
                    function_tool = tool
                    break
            logger.info("No MCP session available, using existing tools only")
            logger.info(f"Total function declarations: {len(function_tool['function_declarations']) if function_tool else 0}")

        # Create the session with updated tools
        session_config = CONFIG.copy()
        session_config["tools"] = tools_config
        
        logger.info("Creating Gemini Live session with MCP tools...")
        session = client.aio.live.connect(model=MODEL, config=session_config)
        logger.info("Gemini Live session created successfully")

        return session, mcp_session

    except Exception as e:
        logger.error(f"Error creating Gemini session: {e}")
        raise
