"""Offline test of Agent mode: a fake model asks for tools, then answers. No API key or network needed."""
import os
from types import SimpleNamespace as NS
from unittest import mock

from streamlit.testing.v1 import AppTest

os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-test-not-real"
calls = []


def fake_create(**kw):
    calls.append(kw)
    usage = NS(total_tokens=10)
    tool_msgs = [m for m in kw["messages"] if isinstance(m, dict) and m.get("role") == "tool"]
    if kw.get("tools") and not tool_msgs:   # first step: ask for two tools
        tc = [NS(id="c1", function=NS(name="calculator", arguments='{"expression": "1250*0.18+40"}'),
                 model_dump=lambda: {"id": "c1", "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "1250*0.18+40"}'}}),
              NS(id="c2", function=NS(name="get_time", arguments='{"place": "Kathmandu"}'),
                 model_dump=lambda: {"id": "c2", "type": "function", "function": {"name": "get_time", "arguments": '{"place": "Kathmandu"}'}})]
        return NS(choices=[NS(message=NS(content="", tool_calls=tc))], usage=usage)
    text = "Tool results were: " + " | ".join(m["content"] for m in tool_msgs) if tool_msgs else "plain reply"
    return NS(choices=[NS(message=NS(content=text, tool_calls=None))], usage=usage)


with mock.patch("openai.OpenAI") as OpenAI:
    OpenAI.return_value.chat.completions.create.side_effect = fake_create
    at = AppTest.from_file("streamlit_app.py", default_timeout=60).run()
    assert not at.exception, at.exception

    def send(text):
        at.session_state.last_sent = 0.0
        at.chat_input[0].set_value(text).run()
        assert not at.exception, at.exception

    toggles = {t.label: t for t in at.sidebar.toggle}
    toggles["AI safety check"].set_value(False)
    send("plain question")
    assert at.session_state.messages[-1]["content"] == "plain reply", at.session_state.messages[-1]
    print("normal mode OK, tools sent:", "tools" in calls[-1])

    {t.label: t for t in at.sidebar.toggle}["🔧 Agent mode (tools)"].set_value(True)
    send("What is 18% of 1250 plus 40, and what time is it in Kathmandu?")
    last = at.session_state.messages[-1]
    print("agent reply:", last["content"])
    print("steps:", [(s["tool"], s["result"]) for s in last["steps"]])
    assert "265.0" in last["content"] and "Asia/Kathmandu" in last["content"]
    assert len(last["steps"]) == 2
    print("tokens:", at.session_state.tokens_used, "| expanders:", [e.label for e in at.expander])

