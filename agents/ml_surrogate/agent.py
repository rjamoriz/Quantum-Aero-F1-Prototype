"""
ML Surrogate Agent
Handles fast ML-based aerodynamic predictions with confidence assessment
"""
import asyncio
import json
from typing import Dict, Any, Optional
import aiohttp
from datetime import datetime

from agents.utils.anthropic_client import claude_client
from agents.utils.nats_client import NATSClient
from agents.config.config import get_agent_config


ML_SURROGATE_AGENT_PROMPT = """
You are the ML Surrogate Agent responsible for fast aerodynamic predictions using trained neural networks.

ROLE:
- Select appropriate ML model for given query
- Run GPU-accelerated inference
- Assess prediction confidence
- Recommend when to escalate to high-fidelity CFD
- Explain model predictions in aerodynamic terms

MODELS AVAILABLE:
1. GeoConvNet-v2.1 - Pressure field prediction (100K meshes, <5% error)
2. ForceNet-v1.3 - Direct Cl/Cd prediction (50K meshes, <2% error)
3. TransientNet-v1.0 - Unsteady flow prediction (10K cases, <10% error)

DECISION LOGIC:
- Use GeoConvNet for detailed pressure analysis
- Use ForceNet for quick design screening
- Use TransientNet for DRS/corner-exit scenarios
- If confidence < 0.9, recommend physics validation
- If geometry outside training distribution, flag uncertainty

CONFIDENCE ASSESSMENT:
Consider these factors:
1. Is the mesh within the training distribution?
2. Are predicted values physically plausible?
3. Are there any anomalies (NaN, extreme values)?
4. How consistent are predictions across similar meshes?
5. What is the mesh quality?

OUTPUT FORMAT:
{
  "prediction": {
    "Cl": 2.45,
    "Cd": 0.385,
    "L_D": 6.36,
    "pressure_field": "..." // if requested
  },
  "confidence": 0.95,
  "model_used": "GeoConvNet-v2.1",
  "inference_time_ms": 67,
  "reasoning": "High confidence because...",
  "concerns": ["List any concerns"],
  "recommendation": "High confidence, no validation needed" | "Physics validation recommended"
}
"""


class MLSurrogateAgent:
    """ML Surrogate Agent for fast aerodynamic predictions"""

    def __init__(self):
        self.config = get_agent_config("ml_surrogate")
        self.nats = NATSClient()
        self.ml_service_url = self.config["services"]["ml_inference"]
        self.running = False

    async def start(self):
        """Start the ML Surrogate Agent"""
        print("=" * 60)
        print("🧠 Starting ML Surrogate Agent")
        print("=" * 60)

        # Connect to NATS
        await self.nats.connect()

        # Subscribe to prediction requests
        await self.nats.subscribe(
            "agent.ml.predict",
            self._handle_prediction_request,
            queue="ml-agent-queue"
        )

        # Subscribe to batch requests
        await self.nats.subscribe(
            "agent.ml.batch",
            self._handle_batch_request,
            queue="ml-agent-queue"
        )

        self.running = True
        print("✓ ML Surrogate Agent is ready")

        # Keep running
        while self.running:
            await asyncio.sleep(1)

    async def stop(self):
        """Stop the agent"""
        self.running = False
        await self.nats.disconnect()
        print("✓ ML Surrogate Agent stopped")

    async def _handle_prediction_request(self, request: Dict[str, Any]):
        """
        Handle prediction request from other agents

        Args:
            request: {
                "mesh_id": str,
                "parameters": dict,
                "model_preference": "GeoConvNet" | "ForceNet" | "auto",
                "return_pressure_field": bool
            }
        """
        try:
            print(f"\n🔮 Prediction request for mesh: {request.get('mesh_id')}")

            mesh_id = request["mesh_id"]
            parameters = request.get("parameters", {})
            model_pref = request.get("model_preference", "auto")
            return_pressure = request.get("return_pressure_field", False)

            # 1. Select appropriate model
            model_name = await self._select_model(request)
            print(f"  📊 Selected model: {model_name}")

            # 2. Call ML service
            start_time = datetime.utcnow()
            ml_result = await self._call_ml_service(
                mesh_id, parameters, model_name, return_pressure
            )
            inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # 3. Assess confidence with Claude
            confidence_assessment = await self._assess_confidence(
                ml_result, mesh_id, parameters
            )

            # 4. Build response
            response = {
                "prediction": ml_result,
                "confidence": confidence_assessment["confidence"],
                "model_used": model_name,
                "inference_time_ms": inference_time,
                "reasoning": confidence_assessment["reasoning"],
                "concerns": confidence_assessment.get("concerns", []),
                "recommendation": (
                    "Physics validation recommended"
                    if confidence_assessment["confidence"] < 0.9
                    else "High confidence, proceed with results"
                ),
                "timestamp": datetime.utcnow().isoformat()
            }

            print(f"  ✓ Prediction complete: Cl={ml_result.get('Cl', 'N/A'):.2f}, confidence={confidence_assessment['confidence']:.2f}")

            # Publish result
            await self.nats.publish("agent.ml.result", response)

            return response

        except Exception as e:
            print(f"  ✗ Prediction failed: {e}")
            error_response = {"error": str(e), "mesh_id": request.get("mesh_id")}
            await self.nats.publish("agent.ml.error", error_response)
            return error_response

    async def _handle_batch_request(self, request: Dict[str, Any]):
        """Handle batch prediction requests"""
        print(f"\n📦 Batch prediction request: {len(request.get('variants', []))} variants")

        variants = request.get("variants", [])
        results = []

        # Process all variants in parallel
        tasks = [self._handle_prediction_request(variant) for variant in variants]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Publish batch result
        await self.nats.publish("agent.ml.batch_result", {
            "results": results,
            "batch_id": request.get("batch_id"),
            "timestamp": datetime.utcnow().isoformat()
        })

        return {"results": results, "batch_id": request.get("batch_id")}

    async def _select_model(self, request: Dict[str, Any]) -> str:
        """
        Select appropriate ML model based on request

        Args:
            request: Prediction request

        Returns:
            Model name
        """
        model_pref = request.get("model_preference", "auto")

        if model_pref != "auto":
            return model_pref

        # Auto selection based on request characteristics
        if request.get("return_pressure_field"):
            return "GeoConvNet-v2.1"
        elif request.get("parameters", {}).get("transient"):
            return "TransientNet-v1.0"
        else:
            return "ForceNet-v1.3"  # Default for quick screening

    async def _call_ml_service(
        self,
        mesh_id: str,
        parameters: Dict[str, Any],
        model_name: str,
        return_pressure: bool
    ) -> Dict[str, Any]:
        """
        Call ML inference microservice

        Args:
            mesh_id: Mesh identifier
            parameters: Flow parameters
            model_name: ML model to use
            return_pressure: Whether to return pressure field

        Returns:
            ML prediction results
        """
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "mesh_id": mesh_id,
                    "model": model_name,
                    "return_pressure_field": return_pressure,
                    **parameters
                }

                async with session.post(
                    f"{self.ml_service_url}/api/v1/predict",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        error_text = await resp.text()
                        raise Exception(f"ML service error: {error_text}")

        except asyncio.TimeoutError:
            raise Exception("ML service timeout")
        except Exception as e:
            raise Exception(f"ML service call failed: {e}")

    async def _assess_confidence(
        self,
        ml_result: Dict[str, Any],
        mesh_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Claude to assess prediction confidence

        Args:
            ml_result: ML prediction results
            mesh_id: Mesh identifier
            parameters: Flow parameters

        Returns:
            Confidence assessment
        """
        messages = [{
            "role": "user",
            "content": f"""Assess the confidence of this ML prediction:

Mesh ID: {mesh_id}
Parameters: {json.dumps(parameters, indent=2)}
Prediction: {json.dumps(ml_result, indent=2)}

Consider:
1. Is the mesh within the training distribution?
2. Are predicted values physically plausible?
3. Are there any anomalies (NaN, extreme values)?
4. How consistent are predictions across similar meshes?

Return JSON:
{{
  "confidence": 0.0-1.0,
  "reasoning": "Explanation of confidence level",
  "concerns": ["List any concerns"]
}}"""
        }]

        try:
            response = await claude_client.create_message_with_retry(
                system=ML_SURROGATE_AGENT_PROMPT,
                messages=messages,
                model=self.config["anthropic"]["model"],
                temperature=0.1,
                max_tokens=1024,
            )

            result = await claude_client.extract_json_from_response(response)
            if result:
                return result
            else:
                # Fallback if JSON extraction fails
                return {
                    "confidence": 0.8,
                    "reasoning": "Default confidence (JSON parsing failed)",
                    "concerns": []
                }

        except Exception as e:
            print(f"  ⚠️  Confidence assessment failed: {e}")
            return {
                "confidence": 0.7,
                "reasoning": f"Confidence assessment failed: {e}",
                "concerns": ["Assessment failed"]
            }


async def main():
    """Example usage of ML Surrogate Agent"""

    # Initialize agent
    agent = MLSurrogateAgent()

    # Start in background
    agent_task = asyncio.create_task(agent.start())

    # Wait for agent to be ready
    await asyncio.sleep(2)

    # Example: Send prediction request via NATS
    print("\n" + "=" * 60)
    print("Example: Prediction request")
    print("=" * 60)

    nats = NATSClient()
    await nats.connect()

    response = await nats.request(
        "agent.ml.predict",
        {
            "mesh_id": "wing_v3.2",
            "parameters": {
                "velocity": 250,
                "yaw": 0,
                "reynolds": 5e6
            },
            "model_preference": "auto",
            "return_pressure_field": False
        },
        timeout=15.0
    )

    print(json.dumps(response, indent=2))

    # Cleanup
    await nats.disconnect()
    await agent.stop()
    agent_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
