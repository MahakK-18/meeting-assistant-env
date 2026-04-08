"""
Meeting Assistant RL Environment - Server Side
An OpenEnv environment where an agent learns to summarize meeting notes
and extract action items, scored by a reward function.
"""

import re
from openenv.core.environment import Environment
from meeting_assistant_env.models import MeetingAction, MeetingObservation, MeetingState


MEETING_CORPUS = [
    {
        "transcript": """
Project sync - Q2 planning. Sarah will finalize the budget report by Thursday.
The team discussed moving the launch date to May 15th due to UI delays.
John needs to fix the authentication bug before the demo next week.
Marketing needs to review the campaign draft. Follow-up meeting scheduled for Friday.
        """,
        "reference_summary": "Q2 planning sync. Launch delayed to May 15 due to UI issues. Auth bug must be fixed before demo.",
        "reference_actions": [
            "Sarah - finalize budget report by Thursday",
            "John - fix authentication bug before demo",
            "Marketing - review campaign draft",
        ],
    },
    {
        "transcript": """
Client call with Acme Corp. They requested a new dashboard feature showing real-time analytics.
Dev team estimates 3 weeks. Priya will send a revised proposal by Monday.
Client also flagged a bug in the export function - Ravi to investigate.
Next check-in call in two weeks.
        """,
        "reference_summary": "Acme Corp requested real-time analytics dashboard (3 weeks est). Bug flagged in export function.",
        "reference_actions": [
            "Priya - send revised proposal by Monday",
            "Ravi - investigate export function bug",
        ],
    },
    {
        "transcript": """
Weekly standup. Backend deployment went smoothly. Frontend team is blocked on API docs.
Neha to share updated API documentation by EOD. Performance testing starts tomorrow.
Team agreed to switch to daily standups until launch. Manager to update calendar invites.
        """,
        "reference_summary": "Backend deployment done. Frontend blocked on API docs. Performance testing starts tomorrow. Moving to daily standups.",
        "reference_actions": [
            "Neha - share API documentation by EOD",
            "Manager - update calendar invites for daily standups",
        ],
    },
    {
        "transcript": """
Design review for the new onboarding flow. UX team presented three options.
Stakeholders preferred option B with minor revisions. Alex to update mockups by Wednesday.
Dev team raised concerns about mobile responsiveness - Lisa to investigate feasibility.
Next design review in one week.
        """,
        "reference_summary": "Onboarding flow design reviewed. Option B selected with revisions. Mobile responsiveness needs investigation.",
        "reference_actions": [
            "Alex - update mockups by Wednesday",
            "Lisa - investigate mobile responsiveness feasibility",
        ],
    },
    {
        "transcript": """
Quarterly review with leadership. Revenue target met at 103%. Customer churn increased by 2%.
Product team to present retention strategy by next Monday. Sales team exceeded quota.
HR to schedule team appreciation event. Budget for Q3 approved with 15% increase.
        """,
        "reference_summary": "Q revenue at 103% of target but churn up 2%. Q3 budget approved with 15% increase.",
        "reference_actions": [
            "Product team - present retention strategy by Monday",
            "HR - schedule team appreciation event",
        ],
    },
]


class MeetingAssistantEnvironment(Environment):
    """
    RL Environment for training agents to summarize meeting notes
    and extract action items from raw transcripts.

    Reward breakdown (max = 1.0):
      - Summary length score   : 0.0–0.30  (concise but not too short)
      - Keyword coverage score : 0.0–0.40  (key decisions captured)
      - Action item score      : 0.0–0.30  (quantity + specificity)
    """

    def __init__(self):
        self._episode_index = 0
        self._current_meeting = None
        self._step_count = 0
        self._done = False

    def reset(self) -> MeetingObservation:
        self._current_meeting = MEETING_CORPUS[self._episode_index % len(MEETING_CORPUS)]
        self._episode_index += 1
        self._step_count = 0
        self._done = False

        return MeetingObservation(
            transcript=self._current_meeting["transcript"].strip(),
            summary=None,
            action_items=[],
            feedback="New meeting transcript loaded. Provide a summary and action items.",
            done=False,
        )

    def step(self, action: MeetingAction) -> tuple[MeetingObservation, float, bool]:
        self._step_count += 1
        self._done = True

        reward = self._compute_reward(action)
        feedback = self._generate_feedback(action, reward)

        observation = MeetingObservation(
            transcript=self._current_meeting["transcript"].strip(),
            summary=action.summary,
            action_items=action.action_items,
            feedback=feedback,
            done=True,
        )
        return observation, reward, True

    def state(self) -> MeetingState:
        return MeetingState(
            episode_index=self._episode_index,
            step_count=self._step_count,
            done=self._done,
        )

    # ------------------------------------------------------------------ #
    #  Reward Function                                                      #
    # ------------------------------------------------------------------ #

    def _compute_reward(self, action: MeetingAction) -> float:
        s1 = self._score_summary_length(action.summary)
        s2 = self._score_keyword_coverage(action.summary)
        s3 = self._score_action_items(action.action_items)
        return round(min(s1 + s2 + s3, 1.0), 4)

    def _score_summary_length(self, summary: str) -> float:
        if not summary:
            return 0.0
        wc = len(summary.split())
        if 20 <= wc <= 80:
            return 0.30
        elif wc < 20:
            return 0.10
        return 0.15  # too verbose

    def _score_keyword_coverage(self, summary: str) -> float:
        if not summary:
            return 0.0
        ref = self._current_meeting["reference_summary"].lower()
        ref_kw = set(re.findall(r'\b\w{4,}\b', ref))
        agent_kw = set(summary.lower().split())
        if not ref_kw:
            return 0.2
        return round(len(ref_kw & agent_kw) / len(ref_kw) * 0.4, 4)

    def _score_action_items(self, items: list[str]) -> float:
        if not items:
            return 0.0
        ref_n = len(self._current_meeting["reference_actions"])
        qty_score = min(len(items) / max(ref_n, 1), 1.0) * 0.15
        deadline_re = re.compile(
            r'\b(by|before|on|eod|monday|tuesday|wednesday|thursday|friday|today|tomorrow)\b', re.I
        )
        specificity = 0.0
        for item in items:
            if len(item.split()) >= 4:
                specificity += 0.03
            if deadline_re.search(item):
                specificity += 0.02
        return round(min(qty_score + specificity, 0.30), 4)

    def _generate_feedback(self, action: MeetingAction, reward: float) -> str:
        if reward >= 0.80:
            quality = "Excellent! Well-structured summary with specific action items."
        elif reward >= 0.55:
            quality = "Good. Try adding deadlines and assignee names to action items."
        elif reward >= 0.30:
            quality = "Partial credit. Ensure summary covers key decisions and is 20–80 words."
        else:
            quality = "Low score. Summary may be missing or too short; action items lack specificity."
        return f"Reward: {reward:.4f} | {quality}"
