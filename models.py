"""
Pydantic models for the Meeting Assistant RL Environment.
Defines the Action, Observation, and State structures used
by both the server (environment) and client.
"""

from typing import Optional
from pydantic import BaseModel, Field
from openenv.core.models import Action, Observation, State


class MeetingAction(Action):
    """
    The agent's response to a meeting transcript.
    It must provide:
      - summary: a concise paragraph summarising key decisions
      - action_items: a list of specific tasks with owners
    """
    summary: str = Field(
        ...,
        description="A concise summary of the meeting (aim for 20–80 words)."
    )
    action_items: list[str] = Field(
        default_factory=list,
        description="List of action items, e.g. 'Alice - submit report by Friday'."
    )


class MeetingObservation(Observation):
    """
    What the agent sees at each step.
    On reset: raw transcript + empty summary/actions.
    After step: transcript + agent's output + grader feedback.
    """
    transcript: str = Field(..., description="Raw meeting transcript the agent must process.")
    summary: Optional[str] = Field(None, description="Agent's summary (populated after step).")
    action_items: list[str] = Field(default_factory=list, description="Agent's extracted action items.")
    feedback: str = Field("", description="Grader feedback string.")
    done: bool = Field(False, description="Whether the episode is complete.")


class MeetingState(State):
    """Tracks episode-level metadata."""
    episode_index: int = Field(0, description="How many episodes have been run.")
    step_count: int = Field(0, description="Steps taken in the current episode.")
    done: bool = Field(False, description="Whether the current episode is finished.")
