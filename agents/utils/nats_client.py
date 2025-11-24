"""
NATS Messaging Client
Handles pub-sub communication between agents
"""
import asyncio
import json
from typing import Dict, Any, Callable, Optional
from nats.aio.client import Client as NATS
from nats.js import JetStreamContext
from agents.config.config import NATS_CONFIG, NATS_SUBJECTS


class NATSClient:
    """NATS messaging client for agent communication"""

    def __init__(self):
        self.nc: Optional[NATS] = None
        self.js: Optional[JetStreamContext] = None
        self._subscriptions = {}

    async def connect(self):
        """Connect to NATS server"""
        try:
            self.nc = NATS()
            await self.nc.connect(
                servers=NATS_CONFIG["servers"],
                max_reconnect_attempts=NATS_CONFIG["max_reconnect_attempts"],
                reconnect_time_wait=NATS_CONFIG["reconnect_time_wait"],
            )

            # Enable JetStream
            self.js = self.nc.jetstream()

            print(f"✓ Connected to NATS at {NATS_CONFIG['servers'][0]}")

        except Exception as e:
            print(f"✗ Failed to connect to NATS: {e}")
            raise

    async def disconnect(self):
        """Disconnect from NATS server"""
        if self.nc:
            await self.nc.drain()
            await self.nc.close()
            print("✓ Disconnected from NATS")

    async def publish(
        self,
        subject: str,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Publish message to a subject

        Args:
            subject: NATS subject
            data: Message data (will be JSON encoded)
            headers: Optional message headers
        """
        if not self.nc:
            raise RuntimeError("NATS not connected")

        try:
            message = json.dumps(data).encode()
            await self.nc.publish(subject, message, headers=headers)

        except Exception as e:
            print(f"✗ Failed to publish to {subject}: {e}")
            raise

    async def subscribe(
        self,
        subject: str,
        callback: Callable[[Dict[str, Any]], None],
        queue: Optional[str] = None
    ):
        """
        Subscribe to a subject with callback

        Args:
            subject: NATS subject pattern
            callback: Async function to handle messages
            queue: Optional queue group name for load balancing
        """
        if not self.nc:
            raise RuntimeError("NATS not connected")

        async def message_handler(msg):
            try:
                data = json.loads(msg.data.decode())
                await callback(data)
            except Exception as e:
                print(f"✗ Error in message handler for {subject}: {e}")

        try:
            sub = await self.nc.subscribe(subject, cb=message_handler, queue=queue)
            self._subscriptions[subject] = sub
            print(f"✓ Subscribed to {subject}" + (f" [queue: {queue}]" if queue else ""))

        except Exception as e:
            print(f"✗ Failed to subscribe to {subject}: {e}")
            raise

    async def request(
        self,
        subject: str,
        data: Dict[str, Any],
        timeout: float = 5.0
    ) -> Dict[str, Any]:
        """
        Request-reply pattern

        Args:
            subject: NATS subject
            data: Request data
            timeout: Request timeout in seconds

        Returns:
            Response data
        """
        if not self.nc:
            raise RuntimeError("NATS not connected")

        try:
            message = json.dumps(data).encode()
            response = await self.nc.request(subject, message, timeout=timeout)
            return json.loads(response.data.decode())

        except asyncio.TimeoutError:
            print(f"✗ Request to {subject} timed out")
            raise
        except Exception as e:
            print(f"✗ Request to {subject} failed: {e}")
            raise

    async def publish_event(self, event_type: str, data: Dict[str, Any]):
        """
        Publish event using predefined subjects

        Args:
            event_type: Event type (simulation_completed, optimization_requested, etc.)
            data: Event data
        """
        subject = NATS_SUBJECTS.get(event_type)
        if not subject:
            raise ValueError(f"Unknown event type: {event_type}")

        await self.publish(subject, {
            "event_type": event_type,
            "data": data,
        })

    async def subscribe_to_events(
        self,
        event_type: str,
        callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Subscribe to specific event type

        Args:
            event_type: Event type to subscribe to
            callback: Async callback function
        """
        subject = NATS_SUBJECTS.get(event_type)
        if not subject:
            raise ValueError(f"Unknown event type: {event_type}")

        async def event_handler(msg_data):
            if msg_data.get("event_type") == event_type:
                await callback(msg_data.get("data"))

        await self.subscribe(subject, event_handler)

    async def broadcast_to_agents(self, data: Dict[str, Any]):
        """
        Broadcast message to all agents

        Args:
            data: Message data
        """
        await self.publish(NATS_SUBJECTS["agent_broadcast"], data)

    async def unsubscribe(self, subject: str):
        """
        Unsubscribe from a subject

        Args:
            subject: NATS subject
        """
        if subject in self._subscriptions:
            await self._subscriptions[subject].unsubscribe()
            del self._subscriptions[subject]
            print(f"✓ Unsubscribed from {subject}")


# Example usage
async def example_usage():
    """Example NATS client usage"""

    # Initialize client
    client = NATSClient()
    await client.connect()

    # Subscribe to events
    async def on_simulation_complete(data):
        print(f"Simulation completed: {data['mesh_id']}")

    await client.subscribe_to_events("simulation_completed", on_simulation_complete)

    # Publish event
    await client.publish_event("simulation_completed", {
        "mesh_id": "wing_v1.0",
        "results": {"Cl": 2.5, "Cd": 0.35}
    })

    # Request-reply
    response = await client.request(
        "agent.ml.predict",
        {"mesh_id": "wing_v1.0", "velocity": 250}
    )
    print(f"ML prediction: {response}")

    # Cleanup
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(example_usage())
