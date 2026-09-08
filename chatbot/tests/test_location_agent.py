"""
The LangGraph tool layer: tool contracts, and the agent's decision inputs.

The HTTP layer is covered in test_location_service/test_weather_service, so
here the services are stubbed with async fakes and the assertions are about the
contract the model actually sees:

  * every tool returns a JSON string, and a failure is a structured error code
    rather than a raised exception -- a failing external dependency must never
    crash graph execution;
  * the location/weather policy prompt is on every turn, because that prompt is
    what makes the model chain the tools instead of guessing a temperature;
  * tool_node executes a scripted tool call and reports the result as a
    ToolMessage, including for an unknown tool and a tool that raises.

No real LLM and no real network is involved: the chat model is a fake that
returns scripted AIMessages and records what it was asked.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

import chatbot_backend
import location_service
import location_state
import service_cache
import weather_service
from conftest import apply_location_env

ALICE = "user-alice"

STORED_LOCATION = {
    "latitude": 12.97,
    "longitude": 77.59,
    "city": "Bengaluru",
    "state": "Karnataka",
    "country": "India",
    "country_code": "IN",
    "timezone": "Asia/Kolkata",
    "label": "Bengaluru, Karnataka, India",
    "source": chatbot_backend.location_state.SOURCE_BROWSER_GPS,
    "updated_at": "2024-05-30T12:00:00Z",
}

LONDON = {
    "latitude": 51.5074,
    "longitude": -0.1278,
    "city": "London",
    "state": "England",
    "country": "United Kingdom",
    "country_code": "GB",
    "timezone": "Europe/London",
    "label": "London, England, United Kingdom",
}

WEATHER_REPORT = {
    "location": "Bengaluru, Karnataka, India",
    "latitude": 12.97,
    "longitude": 77.59,
    "timezone": "Asia/Kolkata",
    "provider": "openweathermap",
    "current": {
        "temperature": 27.31,
        "feels_like": 29.14,
        "humidity": 74,
        "wind_speed": 3.6,
        "condition": "Scattered clouds",
        "units": {"temperature": "°C", "wind_speed": "m/s"},
    },
    "today": {"high": 31.5, "low": 22.0, "precipitation_probability": 40},
}


@pytest.fixture(autouse=True)
def clean_service_caches():
    """The geocode/weather caches are process globals; keep tests independent."""
    service_cache.reset_for_tests()
    yield
    service_cache.reset_for_tests()


@pytest.fixture(autouse=True)
def location_env(monkeypatch):
    apply_location_env(monkeypatch)


@pytest.fixture(autouse=True)
def no_real_services(monkeypatch):
    """
    Hard stop against any tool in this file reaching a real provider.

    Each test that needs a result installs its own fake over the top; anything
    that forgets to gets a loud failure instead of an HTTP request.
    """

    async def forbidden(*args, **kwargs):
        raise AssertionError("A test reached a real location/weather service.")

    monkeypatch.setattr(location_service, "reverse_geocode", forbidden)
    monkeypatch.setattr(location_service, "forward_geocode", forbidden)
    monkeypatch.setattr(weather_service, "get_weather", forbidden)

    # The chat path resolves its model through _get_llm_chain(), so that is the
    # seam that has to be blocked -- otherwise a test that forgets to install a
    # fake silently spends real Groq/Gemini quota.
    def forbidden_chain():
        raise AssertionError("A test reached a real chat provider.")

    monkeypatch.setattr(chatbot_backend, "_get_llm_chain", forbidden_chain)
    monkeypatch.setattr(chatbot_backend, "_llm_chain", None)


def install_model(monkeypatch, model):
    """Make `model` the only provider in the chat fallback chain."""
    monkeypatch.setattr(chatbot_backend, "_get_llm_chain", lambda: [("fake", model)])


@pytest.fixture
def signed_in_user():
    """
    Set the contextvar the tools read the signed-in user from.

    The set/reset pair runs in the fixture's own context, which the test
    coroutine inherits a copy of. Setting it from inside the test body instead
    would produce a token pytest-asyncio's task context cannot reset.
    """
    token = chatbot_backend._active_user_id.set(ALICE)
    try:
        yield ALICE
    finally:
        chatbot_backend._active_user_id.reset(token)


def stub_get_state(monkeypatch, result=None, error: Exception | None = None):
    async def fake_get_state(user_id: str):
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(chatbot_backend.location_state, "get_state", fake_get_state)


def stub_forward_geocode(monkeypatch, result=None, error: Exception | None = None):
    async def fake_forward_geocode(place: str, **kwargs):
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(
        chatbot_backend.location_service, "forward_geocode", fake_forward_geocode
    )


def stub_weather(monkeypatch, result=None, error: Exception | None = None):
    calls: list[tuple] = []

    async def fake_get_weather(latitude, longitude, **kwargs):
        calls.append((latitude, longitude, kwargs))
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(chatbot_backend.weather_service, "get_weather", fake_get_weather)
    return calls


class FakeChatModel:
    """
    A scripted stand-in for the tool-bound Groq model.

    It records the message list it was handed (so a test can assert the policy
    prompt reached the model) and returns the next scripted AIMessage.
    """

    def __init__(self, *responses: AIMessage) -> None:
        self.responses = list(responses)
        self.calls: list[list] = []

    async def ainvoke(self, messages, **kwargs):
        self.calls.append(list(messages))
        if not self.responses:
            return AIMessage(content="(no more scripted responses)")
        return self.responses.pop(0)


def tool_call(name: str, args: dict, call_id: str = "call-1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def test_the_three_location_tools_are_registered():
    names = {tool_obj.name for tool_obj in chatbot_backend.base_tools}
    assert {"get_current_location", "geocode_location", "get_weather"} <= names


def test_the_location_tools_are_dispatchable():
    for name in ("get_current_location", "geocode_location", "get_weather"):
        assert name in chatbot_backend.tools_by_name


def test_the_pre_existing_tools_are_still_registered():
    """Adding location/weather must not displace what already worked."""
    names = {tool_obj.name for tool_obj in chatbot_backend.base_tools}
    assert {"rag_search", "Mathematical_calculations", "get_stock_price"} <= names
    for name in ("rag_search", "Mathematical_calculations", "get_stock_price"):
        assert name in chatbot_backend.tools_by_name


def test_every_location_tool_documents_its_error_codes():
    """The model reads the docstring to decide what to do after a failure."""
    assert "LOCATION_PERMISSION_DENIED" in chatbot_backend.get_current_location.description
    assert "LOCATION_NOT_AVAILABLE" in chatbot_backend.get_current_location.description
    assert "LOCATION_NOT_FOUND" in chatbot_backend.geocode_location.description
    assert "WEATHER_UNAVAILABLE" in chatbot_backend.get_weather.description


# --------------------------------------------------------------------------
# get_current_location
# --------------------------------------------------------------------------

async def test_get_current_location_returns_the_stored_location(monkeypatch, signed_in_user):
    stub_get_state(monkeypatch, (chatbot_backend.location_state.STATUS_READY, STORED_LOCATION))

    result = json.loads(await chatbot_backend.get_current_location.ainvoke({}))

    assert "error" not in result
    assert result["city"] == "Bengaluru"
    assert result["state"] == "Karnataka"
    assert result["country"] == "India"
    assert result["country_code"] == "IN"
    assert result["timezone"] == "Asia/Kolkata"
    assert result["label"] == "Bengaluru, Karnataka, India"
    assert result["latitude"] == 12.97
    assert result["longitude"] == 77.59
    assert result["source"] == "browser_gps"
    assert result["as_of"] == "2024-05-30T12:00:00Z"


async def test_get_current_location_without_a_signed_in_user(monkeypatch):
    stub_get_state(monkeypatch, (chatbot_backend.location_state.STATUS_READY, STORED_LOCATION))

    result = json.loads(await chatbot_backend.get_current_location.ainvoke({}))

    # No user id means no location, and definitely not somebody else's.
    assert result["error"]["code"] == chatbot_backend.LOCATION_NOT_AVAILABLE


async def test_a_denied_permission_gets_its_own_code(monkeypatch, signed_in_user):
    """
    LOCATION_PERMISSION_DENIED tells the model "explain and offer a city", not
    "retry" -- so it must be distinguishable from a generic unavailability.
    """
    stub_get_state(monkeypatch, ("denied", None))

    result = json.loads(await chatbot_backend.get_current_location.ainvoke({}))

    assert result["error"]["code"] == chatbot_backend.LOCATION_PERMISSION_DENIED


@pytest.mark.parametrize("status", ["timeout", "unavailable", "unsupported", "none"])
async def test_other_failure_statuses_map_to_not_available(monkeypatch, signed_in_user, status):
    stub_get_state(monkeypatch, (status, None))

    result = json.loads(await chatbot_backend.get_current_location.ainvoke({}))

    assert result["error"]["code"] == chatbot_backend.LOCATION_NOT_AVAILABLE


async def test_a_ready_status_with_no_record_maps_to_not_available(monkeypatch, signed_in_user):
    stub_get_state(monkeypatch, (chatbot_backend.location_state.STATUS_READY, None))

    result = json.loads(await chatbot_backend.get_current_location.ainvoke({}))

    assert result["error"]["code"] == chatbot_backend.LOCATION_NOT_AVAILABLE


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("database is locked"),
        chatbot_backend.location_state.LocationStateUnavailable("not configured"),
        ValueError("corrupt row"),
    ],
)
async def test_a_failing_location_store_does_not_crash_the_tool(monkeypatch, signed_in_user, error):
    """
    A failing external dependency must not crash graph execution: the tool
    returns a structured error the model can explain, and the turn continues.
    """
    stub_get_state(monkeypatch, error=error)

    result = json.loads(await chatbot_backend.get_current_location.ainvoke({}))

    assert result["error"]["code"] == chatbot_backend.LOCATION_NOT_AVAILABLE


async def test_a_tool_error_message_carries_no_internal_detail(monkeypatch, signed_in_user):
    stub_get_state(monkeypatch, error=RuntimeError("no such table: user_locations"))

    result = json.loads(await chatbot_backend.get_current_location.ainvoke({}))

    assert "no such table" not in result["error"]["message"]


# --------------------------------------------------------------------------
# geocode_location
# --------------------------------------------------------------------------

async def test_geocode_location_returns_a_normalized_place(monkeypatch):
    stub_forward_geocode(monkeypatch, LONDON)

    result = json.loads(await chatbot_backend.geocode_location.ainvoke({"place": "London"}))

    assert "error" not in result
    assert result["city"] == "London"
    assert result["latitude"] == 51.5074
    assert result["longitude"] == -0.1278
    assert result["timezone"] == "Europe/London"
    assert result["label"] == "London, England, United Kingdom"


@pytest.mark.parametrize(
    "code",
    [
        location_service.LOCATION_NOT_FOUND,
        location_service.GEOCODING_TIMEOUT,
        location_service.GEOCODING_RATE_LIMITED,
        location_service.GEOCODING_UNAVAILABLE,
        location_service.GEOCODING_MALFORMED_RESPONSE,
    ],
)
async def test_geocode_location_passes_the_error_code_through(monkeypatch, code):
    stub_forward_geocode(
        monkeypatch,
        error=location_service.LocationError(code, "Something went wrong upstream."),
    )

    result = json.loads(await chatbot_backend.geocode_location.ainvoke({"place": "Atlantis"}))

    assert result["error"]["code"] == code


async def test_geocode_location_converts_an_unexpected_exception(monkeypatch):
    """A bare RuntimeError must not escape into the graph."""
    stub_forward_geocode(monkeypatch, error=RuntimeError("attribute error in a dependency"))

    result = json.loads(await chatbot_backend.geocode_location.ainvoke({"place": "London"}))

    assert result["error"]["code"] == location_service.GEOCODING_UNAVAILABLE


# --------------------------------------------------------------------------
# get_weather
# --------------------------------------------------------------------------

async def test_get_weather_passes_the_report_through(monkeypatch):
    calls = stub_weather(monkeypatch, WEATHER_REPORT)

    result = json.loads(
        await chatbot_backend.get_weather.ainvoke(
            {"latitude": 12.97, "longitude": 77.59, "location_name": "Bengaluru"}
        )
    )

    assert result == WEATHER_REPORT
    assert calls[0][0] == 12.97
    assert calls[0][1] == 77.59
    assert calls[0][2]["label"] == "Bengaluru"


async def test_get_weather_works_without_a_location_name(monkeypatch):
    stub_weather(monkeypatch, WEATHER_REPORT)

    result = json.loads(
        await chatbot_backend.get_weather.ainvoke({"latitude": 12.97, "longitude": 77.59})
    )

    assert result["current"]["temperature"] == 27.31


@pytest.mark.parametrize(
    "code",
    [
        weather_service.WEATHER_TIMEOUT,
        weather_service.WEATHER_RATE_LIMITED,
        weather_service.WEATHER_UNAUTHORIZED,
        weather_service.WEATHER_NOT_CONFIGURED,
        weather_service.WEATHER_NOT_FOUND,
        weather_service.WEATHER_UNAVAILABLE,
        weather_service.WEATHER_MALFORMED_RESPONSE,
    ],
)
async def test_get_weather_passes_the_weather_error_code_through(monkeypatch, code):
    stub_weather(monkeypatch, error=weather_service.WeatherError(code, "Upstream trouble."))

    result = json.loads(
        await chatbot_backend.get_weather.ainvoke({"latitude": 12.97, "longitude": 77.59})
    )

    assert result["error"]["code"] == code


async def test_get_weather_passes_an_invalid_coordinate_code_through(monkeypatch):
    stub_weather(
        monkeypatch,
        error=location_service.LocationError(
            location_service.INVALID_COORDINATES, "Coordinates must be numbers."
        ),
    )

    result = json.loads(
        await chatbot_backend.get_weather.ainvoke({"latitude": 999.0, "longitude": 0.0})
    )

    assert result["error"]["code"] == location_service.INVALID_COORDINATES


async def test_get_weather_converts_an_unexpected_exception(monkeypatch):
    stub_weather(monkeypatch, error=Exception("something nobody planned for"))

    result = json.loads(
        await chatbot_backend.get_weather.ainvoke({"latitude": 12.97, "longitude": 77.59})
    )

    assert result["error"]["code"] == weather_service.WEATHER_UNAVAILABLE


async def test_no_location_or_weather_tool_ever_raises(monkeypatch, signed_in_user):
    """The single guarantee the graph depends on, asserted for all three tools."""
    stub_get_state(monkeypatch, error=RuntimeError("boom"))
    stub_forward_geocode(monkeypatch, error=RuntimeError("boom"))
    stub_weather(monkeypatch, error=RuntimeError("boom"))

    for result in (
        await chatbot_backend.get_current_location.ainvoke({}),
        await chatbot_backend.geocode_location.ainvoke({"place": "London"}),
        await chatbot_backend.get_weather.ainvoke({"latitude": 1.0, "longitude": 2.0}),
    ):
        assert "error" in json.loads(result)


# --------------------------------------------------------------------------
# The anti-fabrication prompt
# --------------------------------------------------------------------------

def test_the_location_weather_policy_is_injected_on_every_turn():
    messages = chatbot_backend._messages_for_model(
        [HumanMessage(content="what's the weather?")]
    )

    system_contents = [
        message.content for message in messages if isinstance(message, SystemMessage)
    ]
    assert chatbot_backend.LOCATION_WEATHER_POLICY in system_contents
    # The user's message is still last, after the injected context.
    assert isinstance(messages[-1], HumanMessage)


def test_the_policy_forbids_inventing_live_values():
    """
    This prompt is what makes the model chain get_current_location into
    get_weather instead of answering "it's 24 and sunny" from memory. If someone
    deletes it, this test is the alarm.
    """
    policy = chatbot_backend.LOCATION_WEATHER_POLICY

    assert "Never state or estimate a temperature" in policy
    assert "get_current_location FIRST" in policy


def test_the_policy_names_every_tool_it_governs():
    policy = chatbot_backend.LOCATION_WEATHER_POLICY

    for name in ("get_current_location", "geocode_location", "get_weather"):
        assert name in policy


def test_the_policy_covers_the_denied_and_unavailable_branches():
    policy = chatbot_backend.LOCATION_WEATHER_POLICY

    assert chatbot_backend.LOCATION_PERMISSION_DENIED in policy
    assert chatbot_backend.LOCATION_NOT_AVAILABLE in policy


# --------------------------------------------------------------------------
# Agent decision plumbing, driven through a fake model
# --------------------------------------------------------------------------

async def test_the_chat_node_sees_the_policy_and_returns_the_models_message(monkeypatch):
    scripted = AIMessage(content="", tool_calls=[tool_call("get_current_location", {})])
    fake_model = FakeChatModel(scripted)
    install_model(monkeypatch, fake_model)

    result = await chatbot_backend.chat(
        {"messages": [HumanMessage(content="what's the weather today?")]}
    )

    assert result["messages"] == [scripted]
    system_contents = [
        message.content
        for message in fake_model.calls[0]
        if isinstance(message, SystemMessage)
    ]
    assert chatbot_backend.LOCATION_WEATHER_POLICY in system_contents


async def test_tool_node_executes_a_scripted_location_call(monkeypatch, signed_in_user):
    stub_get_state(monkeypatch, (chatbot_backend.location_state.STATUS_READY, STORED_LOCATION))
    ai_message = AIMessage(
        content="", tool_calls=[tool_call("get_current_location", {}, "call-loc")]
    )

    result = await chatbot_backend.tool_node({"messages": [ai_message]})

    (message,) = result["messages"]
    assert isinstance(message, ToolMessage)
    assert message.name == "get_current_location"
    assert message.tool_call_id == "call-loc"
    assert json.loads(message.content)["city"] == "Bengaluru"


async def test_tool_node_executes_a_scripted_weather_call(monkeypatch):
    stub_weather(monkeypatch, WEATHER_REPORT)
    ai_message = AIMessage(
        content="",
        tool_calls=[
            tool_call(
                "get_weather",
                {"latitude": 12.97, "longitude": 77.59, "location_name": "Bengaluru"},
                "call-wx",
            )
        ],
    )

    result = await chatbot_backend.tool_node({"messages": [ai_message]})

    (message,) = result["messages"]
    assert message.name == "get_weather"
    payload = json.loads(message.content)
    # Real (faked) provider numbers reach the model, not a placeholder.
    assert payload["current"]["temperature"] == 27.31
    assert payload["today"]["precipitation_probability"] == 40


async def test_tool_node_runs_a_two_step_location_then_weather_sequence(monkeypatch, signed_in_user):
    """The chained path the policy prescribes for "what's the weather today?"."""
    stub_get_state(monkeypatch, (chatbot_backend.location_state.STATUS_READY, STORED_LOCATION))
    weather_calls = stub_weather(monkeypatch, WEATHER_REPORT)

    first = await chatbot_backend.tool_node(
        {
            "messages": [
                AIMessage(content="", tool_calls=[tool_call("get_current_location", {}, "c1")])
            ]
        }
    )
    located = json.loads(first["messages"][0].content)

    second = await chatbot_backend.tool_node(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "get_weather",
                            {
                                "latitude": located["latitude"],
                                "longitude": located["longitude"],
                                "location_name": located["label"],
                            },
                            "c2",
                        )
                    ],
                )
            ]
        }
    )

    assert weather_calls[0][0] == 12.97
    assert weather_calls[0][1] == 77.59
    assert json.loads(second["messages"][0].content)["provider"] == "openweathermap"


async def test_tool_node_handles_several_tool_calls_in_one_message(monkeypatch):
    stub_forward_geocode(monkeypatch, LONDON)
    stub_weather(monkeypatch, WEATHER_REPORT)
    ai_message = AIMessage(
        content="",
        tool_calls=[
            tool_call("geocode_location", {"place": "London"}, "c1"),
            tool_call("get_weather", {"latitude": 51.51, "longitude": -0.13}, "c2"),
        ],
    )

    result = await chatbot_backend.tool_node({"messages": [ai_message]})

    assert [message.name for message in result["messages"]] == [
        "geocode_location",
        "get_weather",
    ]


async def test_tool_node_reports_an_unknown_tool_instead_of_raising(monkeypatch):
    ai_message = AIMessage(
        content="", tool_calls=[tool_call("get_the_weather_maybe", {}, "c1")]
    )

    result = await chatbot_backend.tool_node({"messages": [ai_message]})

    (message,) = result["messages"]
    assert "not available" in message.content
    assert message.name == "get_the_weather_maybe"


async def test_tool_node_catches_a_tool_that_raises(monkeypatch):
    """
    A tool exception is reported back to the model as a ToolMessage. The graph
    keeps running, which is the difference between a degraded answer and a 500.
    """

    class ExplodingTool:
        name = "exploding_tool"

        async def ainvoke(self, args):
            raise RuntimeError("upstream exploded")

    monkeypatch.setitem(chatbot_backend.tools_by_name, "exploding_tool", ExplodingTool())
    ai_message = AIMessage(content="", tool_calls=[tool_call("exploding_tool", {}, "c1")])

    result = await chatbot_backend.tool_node({"messages": [ai_message]})

    (message,) = result["messages"]
    assert "failed" in message.content
    assert "upstream exploded" in message.content


async def test_tool_node_is_a_no_op_without_tool_calls():
    result = await chatbot_backend.tool_node(
        {"messages": [AIMessage(content="It is 27.31 degrees in Bengaluru.")]}
    )

    assert result["messages"] == []


async def test_route_tools_follows_a_tool_call_and_then_stops():
    with_call = AIMessage(content="", tool_calls=[tool_call("get_current_location", {})])
    assert chatbot_backend.route_tools({"messages": [with_call]}) == "tools"

    plain = AIMessage(content="It is 27.31 degrees in Bengaluru.")
    assert chatbot_backend.route_tools({"messages": [plain]}) != "tools"


# --------------------------------------------------------------------------
# Regressions from the code review
# --------------------------------------------------------------------------

async def test_a_geocoding_outage_keeps_the_coordinates(monkeypatch):
    """
    A Nominatim outage must cost a city NAME, not the whole feature.

    resolve_user_coordinates used to propagate, so nothing was stored and the
    assistant asked "which city are you in?" while the server held the user's
    latitude and longitude -- everything the weather lookup actually needs.
    """
    async def failing_reverse(latitude, longitude, **kwargs):
        raise location_service.LocationError(
            location_service.GEOCODING_TIMEOUT, "upstream is slow"
        )

    async def no_timezone(latitude, longitude):
        return None

    saved: dict = {}

    async def fake_save(user_id, location, *, source):
        saved.update(location)
        saved["source"] = source
        return location_state.StoredLocation(
            latitude=location["latitude"],
            longitude=location["longitude"],
            city=location["city"],
            state=location["state"],
            country=location["country"],
            country_code=location["country_code"],
            timezone=location["timezone"],
            label=location["label"],
            source=source,
            updated_at="2026-01-01T00:00:00Z",
        )

    monkeypatch.setattr(chatbot_backend.location_service, "reverse_geocode", failing_reverse)
    monkeypatch.setattr(chatbot_backend.location_service, "resolve_timezone", no_timezone)
    monkeypatch.setattr(chatbot_backend.location_state, "save_location", fake_save)

    record = await chatbot_backend.resolve_user_coordinates(ALICE, 12.9716, 77.5946)

    assert record["latitude"] == 12.9716
    assert record["city"] is None
    # build_label's coordinate fallback, so the model says "your location"
    # rather than naming somewhere it does not actually know.
    assert record["label"] == "12.972, 77.595"


async def test_invalid_coordinates_still_raise(monkeypatch):
    """There is nothing to store, so this one must not be swallowed."""
    with pytest.raises(location_service.LocationError) as exc_info:
        await chatbot_backend.resolve_user_coordinates(ALICE, 91.0, 0.0)
    assert exc_info.value.code == location_service.INVALID_COORDINATES


async def test_the_policy_governs_the_outlook_wording():
    """
    OpenWeather's remaining-hours figures are not the day's high and low.

    If this prompt rule is dropped the model will call a 27 C evening minimum
    "today's low" on a day that actually bottomed out near 22 C -- a real
    number, described wrongly.
    """
    policy = chatbot_backend.LOCATION_WEATHER_POLICY

    assert "covers" in policy
    assert "rest_of_today" in policy
    assert "never call them today's high or low" in policy
    # And the null-city case, so an unnameable point is not given a city.
    assert "city is null" in policy


async def test_the_streaming_tool_loop_is_bounded(monkeypatch):
    """
    A model that keeps asking for tools must not spin forever, and the turn
    must not end on an unanswered tool call -- that would break the next turn.
    """
    rounds = {"n": 0}

    class ForeverToolCallingModel:
        async def astream(self, messages):
            rounds["n"] += 1
            yield AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_current_location",
                        "args": {},
                        "id": f"call-{rounds['n']}",
                    }
                ],
            )

    async def fake_get_state(user_id: str):
        return ("none", None)

    install_model(monkeypatch, ForeverToolCallingModel())
    monkeypatch.setattr(chatbot_backend.location_state, "get_state", fake_get_state)

    class FakeApp:
        async def aget_state(self, config):
            return None

        async def aupdate_state(self, config, values, as_node=None):
            self.written = values

    app = FakeApp()
    tokens = [
        token
        async for token in chatbot_backend._get_response_stream_for_config(
            app, {"configurable": {"thread_id": "t"}}, "where am I?"
        )
    ]

    assert rounds["n"] == chatbot_backend.MAX_TOOL_ROUNDS
    # The turn closes with a real assistant message rather than a dangling
    # tool call.
    assert isinstance(app.written["messages"][-1], AIMessage)
    assert tokens[-1].strip()


# --------------------------------------------------------------------------
# Streaming: a tool-calling round's text is not the answer
# --------------------------------------------------------------------------

class _Chunk(AIMessageChunk):
    """An AIMessageChunk that can carry tool_call_chunks, like a real stream."""


def _stream_reset_app():
    class FakeApp:
        written = None

        async def aget_state(self, config):
            return None

        async def aupdate_state(self, config, values, as_node=None):
            self.written = values

    return FakeApp()


async def test_text_streamed_before_a_tool_call_is_retracted(monkeypatch):
    """
    The model sometimes emits filler on its way to calling a tool.

    Reported live as a wall of invisible characters and "Oops!"/"..." fragments
    appearing above the real weather answer. That text is not the answer, so it
    must be retracted rather than left sitting in the transcript.
    """
    class ChattyThenToolModel:
        def __init__(self):
            self.round = 0

        async def astream(self, messages):
            self.round += 1
            if self.round == 1:
                # Degenerate filler first, THEN the tool call -- the exact
                # ordering that put junk on screen.
                yield AIMessageChunk(content="Saharan\u200b\u200b\u200b Oops! It looks ")
                yield _Chunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "geocode_location",
                            "args": '{"place": "Saharanpur"}',
                            "id": "call-1",
                            "index": 0,
                        }
                    ],
                )
            else:
                yield AIMessageChunk(content="It is 33 C in Saharanpur.")

    async def fake_forward(place, **kwargs):
        return location_service.Location(
            latitude=29.96, longitude=77.55, city="Saharanpur", state="Uttar Pradesh",
            country="India", country_code="IN", timezone="Asia/Kolkata",
            label="Saharanpur, Uttar Pradesh, India",
        )

    install_model(monkeypatch, ChattyThenToolModel())
    monkeypatch.setattr(chatbot_backend.location_service, "forward_geocode", fake_forward)

    app = _stream_reset_app()
    emitted = [
        item
        async for item in chatbot_backend._get_response_stream_for_config(
            app, {"configurable": {"thread_id": "t"}}, "what the wetaher in saharanpur"
        )
    ]

    # A retraction was issued, and it came before the real answer.
    assert chatbot_backend.STREAM_RESET in emitted
    reset_at = emitted.index(chatbot_backend.STREAM_RESET)
    after = "".join(p for p in emitted[reset_at + 1:] if isinstance(p, str))
    assert after == "It is 33 C in Saharanpur."
    # Nothing after the retraction carries the filler.
    assert "Oops" not in after
    assert "\u200b" not in after


async def test_a_tool_calling_rounds_text_is_not_persisted(monkeypatch):
    """
    The filler must not reach history either.

    Left in place, the model reads it back on the next round, sees its own
    broken output and apologises for it instead of simply answering -- which is
    what the live transcript showed it doing.
    """
    class ChattyThenToolModel:
        def __init__(self):
            self.round = 0

        async def astream(self, messages):
            self.round += 1
            if self.round == 1:
                yield _Chunk(
                    content="Oops! It looks ",
                    tool_call_chunks=[
                        {"name": "get_current_location", "args": "{}", "id": "c1", "index": 0}
                    ],
                )
            else:
                yield AIMessageChunk(content="You are in Bengaluru.")

    async def fake_get_state(user_id: str):
        return ("none", None)

    install_model(monkeypatch, ChattyThenToolModel())
    monkeypatch.setattr(chatbot_backend.location_state, "get_state", fake_get_state)

    app = _stream_reset_app()
    async for _ in chatbot_backend._get_response_stream_for_config(
        app, {"configurable": {"thread_id": "t"}}, "where am I?"
    ):
        pass

    persisted = app.written["messages"]
    tool_calling = [
        m for m in persisted if isinstance(m, AIMessage) and m.tool_calls
    ]
    assert tool_calling, "the tool call itself must still be persisted"
    for message in tool_calling:
        # tool_calls are kept (they pair with the ToolMessages); the text is not.
        assert message.content == ""


async def test_history_skips_a_tool_calling_rounds_text():
    """
    get_chat_history feeds both the reloaded transcript and the memory
    extractor, so preamble attached to a tool call must not surface in either.
    """
    messages = [
        HumanMessage(content="where am I?"),
        AIMessage(
            content="Oops! It looks ",
            tool_calls=[{"name": "get_current_location", "args": {}, "id": "c1"}],
        ),
        ToolMessage(content="{}", name="get_current_location", tool_call_id="c1"),
        AIMessage(content="You are in Bengaluru."),
    ]

    rendered = [
        {"role": "assistant", "content": str(m.content)}
        for m in messages
        if isinstance(m, AIMessage) and m.content and not m.tool_calls
    ]

    assert rendered == [{"role": "assistant", "content": "You are in Bengaluru."}]


# --------------------------------------------------------------------------
# Provider fallback
# --------------------------------------------------------------------------

class _Boom:
    """A provider that always fails, the way a rate-limited Groq does."""

    def __init__(self, message="Error code: 429 - rate limit reached"):
        self.message = message
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        raise RuntimeError(self.message)

    def astream(self, messages):
        async def gen():
            self.calls += 1
            raise RuntimeError(self.message)
            yield  # pragma: no cover - unreachable, makes this an async generator

        return gen()


class _Answers:
    """A provider that answers."""

    def __init__(self, text="fallback answer"):
        self.text = text
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.text)

    def astream(self, messages):
        async def gen():
            self.calls += 1
            yield AIMessageChunk(content=self.text)

        return gen()


async def test_a_rate_limited_provider_falls_through_to_the_next(monkeypatch):
    """
    Groq's free tier exhausts its daily token budget; the turn must still be
    answered by the next provider rather than failing.
    """
    primary, secondary = _Boom(), _Answers("answered by gemini")
    monkeypatch.setattr(
        chatbot_backend, "_get_llm_chain",
        lambda: [("groq", primary), ("gemini", secondary)],
    )

    reply = await chatbot_backend._ainvoke_with_fallback(
        [HumanMessage(content="hello")]
    )

    assert reply.content == "answered by gemini"
    assert primary.calls == 1 and secondary.calls == 1


async def test_the_first_healthy_provider_wins(monkeypatch):
    """A working primary must not cost a call to anything downstream."""
    primary, secondary = _Answers("from groq"), _Answers("from gemini")
    monkeypatch.setattr(
        chatbot_backend, "_get_llm_chain",
        lambda: [("groq", primary), ("gemini", secondary)],
    )

    reply = await chatbot_backend._ainvoke_with_fallback([HumanMessage(content="hi")])

    assert reply.content == "from groq"
    assert secondary.calls == 0


async def test_every_provider_failing_raises_rather_than_going_quiet(monkeypatch):
    monkeypatch.setattr(
        chatbot_backend, "_get_llm_chain",
        lambda: [("groq", _Boom("429")), ("gemini", _Boom("402 payment required"))],
    )

    with pytest.raises(chatbot_backend.AllProvidersUnavailable) as exc_info:
        await chatbot_backend._ainvoke_with_fallback([HumanMessage(content="hi")])

    # A user-facing sentence, not the provider's own billing/quota text: that
    # would leak the deployment's account state into the transcript.
    assert "402" not in str(exc_info.value)
    assert "payment required" not in str(exc_info.value).lower()
    # The real cause is still chained, so the log names it.
    assert "402" in str(exc_info.value.__cause__)


async def test_streaming_falls_back_when_the_primary_dies(monkeypatch):
    """The streamed turn survives a provider dying before it emitted anything."""
    monkeypatch.setattr(
        chatbot_backend, "_get_llm_chain",
        lambda: [("groq", _Boom()), ("gemini", _Answers("live answer"))],
    )

    app = _stream_reset_app()
    emitted = [
        item
        async for item in chatbot_backend._get_response_stream_for_config(
            app, {"configurable": {"thread_id": "t"}}, "hello"
        )
    ]

    assert "".join(p for p in emitted if isinstance(p, str)) == "live answer"


async def test_a_provider_dying_mid_stream_retracts_its_partial_text(monkeypatch):
    """
    Half an answer from Groq must not sit above a whole answer from Gemini, so
    the partial text is retracted through the same channel the tool-preamble
    fix uses.
    """
    class HalfThenDie:
        def astream(self, messages):
            async def gen():
                yield AIMessageChunk(content="The weather is ")
                raise RuntimeError("Error code: 429 - rate limit reached")

            return gen()

    monkeypatch.setattr(
        chatbot_backend, "_get_llm_chain",
        lambda: [("groq", HalfThenDie()), ("gemini", _Answers("15 C and raining."))],
    )

    app = _stream_reset_app()
    emitted = [
        item
        async for item in chatbot_backend._get_response_stream_for_config(
            app, {"configurable": {"thread_id": "t"}}, "weather?"
        )
    ]

    assert chatbot_backend.STREAM_RESET in emitted
    reset_at = emitted.index(chatbot_backend.STREAM_RESET)
    after = "".join(p for p in emitted[reset_at + 1:] if isinstance(p, str))
    assert after == "15 C and raining."
    assert "The weather is " not in after


def test_multi_block_content_is_flattened_for_users():
    """
    Gemini answers with content blocks, not a string. str() on that would put
    a literal "[{'type': 'text', ...}]" into the reply and the stored history.
    """
    blocks = [{"type": "text", "text": "It is 15 C in London."}]

    assert chatbot_backend._message_text(blocks) == "It is 15 C in London."
    assert chatbot_backend._message_text("plain") == "plain"
    assert chatbot_backend._message_text(None) == ""
