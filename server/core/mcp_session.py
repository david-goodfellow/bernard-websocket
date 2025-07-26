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
MCP (Model Context Protocol) session management for Gemini Live API
Following the Google AI MCP integration guide
"""

import asyncio
import json
import logging
import aiohttp
from typing import Optional, Dict, Any
from google.genai import types

logger = logging.getLogger(__name__)

class MCPSession:
    """
    MCP session wrapper that implements the interface expected by Gemini
    Following the Google AI MCP integration guide
    """
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None
        
    async def initialize(self):
        """Initialize the MCP session"""
        try:
            self.session = aiohttp.ClientSession()
            logger.info(f"MCP session initialized with server: {self.server_url}")
        except Exception as e:
            logger.error(f"Failed to initialize MCP session: {e}")
            raise
            
    async def close(self):
        """Close the MCP session"""
        if self.session:
            await self.session.close()
            logger.info("MCP session closed")
            
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool using JSON-RPC 2.0 format"""
        try:
            # Format as JSON-RPC 2.0 request with correct MCP structure
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name, # "get_neo4j_schema",
                    "arguments": arguments # {}
                }
            }
            
            # Make HTTP request to MCP server with proper headers
            headers = {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "mcp-session-id": "test-session"
            }
            
            async with self.session.post(
                f"{self.server_url}/api/mcp",
                json=jsonrpc_request,
                headers=headers
            ) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    
                    if 'text/event-stream' in content_type:
                        # Handle Server-Sent Events (SSE) response
                        result_data = {}
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                data = line[6:]  # Remove 'data: ' prefix
                                if data and data != '[DONE]':
                                    try:
                                        json_data = json.loads(data)
                                        if 'result' in json_data:
                                            result_data = json_data['result']
                                            break
                                        elif 'error' in json_data:
                                            logger.error(f"MCP JSON-RPC error: {json_data['error']}")
                                            return {"error": json_data["error"]}
                                    except json.JSONDecodeError:
                                        continue
                        return result_data
                    else:
                        # Handle regular JSON response
                        response_data = await response.json()
                        if "result" in response_data:
                            return response_data["result"]
                        elif "error" in response_data:
                            logger.error(f"MCP JSON-RPC error: {response_data['error']}")
                            return {"error": response_data["error"]}
                        else:
                            return response_data
                else:
                    error_text = await response.text()
                    logger.error(f"MCP tool call failed: {error_text}")
                    return {"error": error_text}
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return {"error": str(e)}

    async def list_tools(self) -> Dict[str, Any]:
        """List available tools on the MCP server"""
        try:
            # Use the tools/list method to discover available tools
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }
            
            headers = {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "mcp-session-id": "test-session"
            }
            
            async with self.session.post(
                f"{self.server_url}/api/mcp",
                json=jsonrpc_request,
                headers=headers
            ) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    
                    if 'text/event-stream' in content_type:
                        # Handle Server-Sent Events (SSE) response
                        result_data = {}
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                data = line[6:]  # Remove 'data: ' prefix
                                if data and data != '[DONE]':
                                    try:
                                        json_data = json.loads(data)
                                        if 'result' in json_data:
                                            result_data = json_data['result']
                                            break
                                        elif 'error' in json_data:
                                            logger.error(f"MCP JSON-RPC error: {json_data['error']}")
                                            return {"error": json_data["error"]}
                                    except json.JSONDecodeError:
                                        continue
                        return result_data
                    else:
                        # Handle regular JSON response
                        response_data = await response.json()
                        if "result" in response_data:
                            return response_data["result"]
                        elif "error" in response_data:
                            logger.error(f"MCP JSON-RPC error: {response_data['error']}")
                            return {"error": response_data["error"]}
                        else:
                            return response_data
                else:
                    error_text = await response.text()
                    logger.error(f"MCP list tools failed: {error_text}")
                    return {"error": error_text}
        except Exception as e:
            logger.error(f"Error listing MCP tools: {e}")
            return {"error": str(e)}

def create_mcp_tool(mcp_session: MCPSession) -> Dict[str, Any]:
    """
    Create a Gemini tool from the MCP session
    Following the Google AI MCP integration guide
    """
    # Define the MCP tool function declaration as a dictionary (not types.Tool)
    mcp_tool = {
        "function_declarations": [
            {
                "name": "mcp_query",
                "description": "Query the MCP (Model Context Protocol) server for data and information. This tool can access databases, APIs, and other external data sources through the MCP protocol.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "The specific MCP tool to call (e.g., 'cypher_query', 'get_data', etc.)",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Additional parameters for the MCP tool call",
                        }
                    },
                    "required": ["tool_name", "parameters"],
                },
            },
            {
                "name": "mcp_list_tools",
                "description": "List all available tools and functionalities on the MCP (Model Context Protocol) server. Use this to discover what tools are available before calling specific tools.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
        ]
    }
    
    return mcp_tool

async def create_mcp_session() -> Optional[MCPSession]:
    """
    Create an MCP session following the Google AI guide pattern.
    Returns an MCPSession that can be passed directly to Gemini as a tool.
    """
    try:
        server_url = "https://mcp-neo4j-cypher-4vi4raqpfa-uc.a.run.app/api/mcp"
        mcp_session = MCPSession(server_url)
        await mcp_session.initialize()
        logger.info("MCP session created successfully")
        return mcp_session
    except Exception as e:
        logger.error(f"Failed to create MCP session: {e}")
        return None 