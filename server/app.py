from openenv.core.env_server import create_app
from meeting_assistant_env.models import MeetingAction, MeetingObservation
from meeting_assistant_env.server.meeting_environment import MeetingEnvironment

env = MeetingEnvironment(max_turns=3)
app = create_app(env, MeetingAction, MeetingObservation)
