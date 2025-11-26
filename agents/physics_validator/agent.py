"""
Physics Validator Agent
Validates ML predictions using physics-based methods (VLM, Panel)
"""
import asyncio
import json
from typing import Dict, Any
from datetime import datetime

from agents.utils.anthropic_client import claude_client
from agents.utils.nats_client import NATSClient
from agents.config.config import get_agent_config


PHYSICS_VALIDATOR_PROMPT = """
You are the Physics Validator Agent responsible for validating ML predictions with physics-based methods.

EXPERTISE:
- Vortex Lattice Method (VLM)
- Panel methods
- CFD validation
- Error analysis
- Uncertainty quantification

VALIDATION STRATEGY:
1. Compare ML prediction with VLM results
2. Check for physical plausibility
3. Identify discrepancies
4. Assess confidence levels
5. Recommend escalation to high-fidelity if needed

VALIDATION CRITERIA:
- Cl error < 5%: High confidence
- Cl error 5-10%: Medium confidence, acceptable
- Cl error > 10%: Low confidence, escalate to CFD
- Physical plausibility: No negative lift at positive AoA

OUTPUT FORMAT:
1. **Validation Summary** - Pass/Fail with confidence
2. **Error Analysis** - Detailed comparison
3. **Physical Checks** - Plausibility assessment
4. **Recommendation** - Accept/Reject/Escalate
"""


class PhysicsValidatorAgent:
    """Physics validator agent"""

    def __init__(self):
        self.config = get_agent_config("physics_validator")
        self.nats = NATSClient()
        self.running = False

    async def start(self):
        """Start the agent"""
        print("=" * 60)
        print("🔬 Starting Physics Validator Agent")
        print("=" * 60)

        await self.nats.connect()
        await self.nats.subscribe("agent.physics.validate", self._handle_validation)

        self.running = True
        print("✓ Physics Validator Agent is ready")

    async def stop(self):
        """Stop the agent"""
        self.running = False
        await self.nats.disconnect()

    async def _handle_validation(self, msg):
        """Handle validation request"""
        try:
            data = json.loads(msg.data.decode())
            print(f"\n🔬 Validating prediction for: {data.get('mesh_id')}")

            result = await self.validate_prediction(
                ml_prediction=data.get('ml_prediction'),
                mesh_id=data.get('mesh_id'),
                conditions=data.get('conditions')
            )

            await msg.respond(json.dumps(result).encode())

        except Exception as e:
            print(f"✗ Validation failed: {e}")
            await msg.respond(json.dumps({"error": str(e)}).encode())

    async def validate_prediction(
        self,
        ml_prediction: Dict[str, Any],
        mesh_id: str,
        conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate ML prediction with physics"""
        
        # Run VLM validation
        vlm_result = await self._run_vlm_validation(mesh_id, conditions)

        # Compare with Claude
        messages = [{
            "role": "user",
            "content": f"""Validate this ML prediction:

ML Prediction: {json.dumps(ml_prediction, indent=2)}
VLM Result: {json.dumps(vlm_result, indent=2)}
Conditions: {json.dumps(conditions, indent=2)}

Assess:
1. Error magnitude
2. Physical plausibility
3. Confidence level
4. Recommendation
"""
        }]

        response = await claude_client.create_message(
            system=PHYSICS_VALIDATOR_PROMPT,
            messages=messages,
            model=self.config["anthropic"]["model"],
            temperature=0.1,
            max_tokens=1024
        )

        return {
            "ml_prediction": ml_prediction,
            "vlm_result": vlm_result,
            "validation": response["content"][0]["text"],
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _run_vlm_validation(self, mesh_id: str, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Run VLM for validation"""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['physics_service_url']}/api/v1/vlm-solve",
                    json={"mesh_id": mesh_id, **conditions},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {"error": f"VLM failed: {resp.status}"}
        except Exception as e:
            return {"error": str(e)}


async def main():
    agent = PhysicsValidatorAgent()
    await agent.start()

    result = await agent.validate_prediction(
        ml_prediction={"Cl": 2.8, "Cd": 0.42, "confidence": 0.85},
        mesh_id="wing_v3.2",
        conditions={"velocity": 250, "alpha": 5.0}
    )

    print(json.dumps(result, indent=2))
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
