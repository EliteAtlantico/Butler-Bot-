"""An LLM that runs the whole BracketBot through tool calls.

    python -m robot_agent "put the remote in the basket, then come back to the sofa"
    python -m robot_agent --voice --viewer

`tools.RobotTools` owns one simulated robot in the living room and exposes
everything the team's code can do -- chores and picks (Hand_and_Wrists),
navigation and search (main_mujoco, comp_vision_sim), base motion, arm joints,
grippers and the cameras -- as OpenAI-style function tools. `agent.RobotAgent`
is the loop: the local model calls a tool, sees how it went, and decides what
to do next until it can answer.
"""
