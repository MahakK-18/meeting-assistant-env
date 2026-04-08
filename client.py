"""
Client for the Meeting Assistant OpenEnv environment.
Use this to connect to a running instance of the environment server.

Usage (sync):
    from meeting_assistant_env.client import MeetingAssistantEnv, MeetingAction

    with MeetingAssistantEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset()
        print(obs.transcript)

        result = env.step(MeetingAction(
            summary="Team agreed to delay launch. Bug fixes prioritised.",
            action_items=["John - fix auth bug by Friday", "Sarah - update timeline by EOD"]
        ))
        print(result.observation.feedback)
        print("Reward:", result.reward)

Usage (async):
    import asyncio
    from meeting_assistant_env.client import MeetingAssistantEnv, MeetingAction

    async def main():
        async with MeetingAssistantEnv(base_url="http://localhost:8000") as env:
            obs = await env.reset()
            result = await env.step(MeetingAction(
                summary="...",
                action_items=["..."]
            ))
            print(result.reward)

    asyncio.run(main())
"""

from openenv.core.env_client import EnvClient
from meeting_assistant_env.models import MeetingAction, MeetingObservation


class MeetingAssistantEnv(EnvClient):
    """
    OpenEnv client for the Meeting Assistant environment.
    Connects to a running server via WebSocket.
    """
    action_type = MeetingAction
    observation_type = MeetingObservation
