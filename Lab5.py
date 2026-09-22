import json
import requests
import streamlit as st
from openai import OpenAI

# --- Setup ---
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY not found in secrets.")
    st.stop()

client = OpenAI(api_key=openai_api_key)
MODEL = "gpt-4o-mini"
DEFAULT_LOCATION = "Syracuse, NY"

st.title("Lab 5 – The 'What to Wear' Bot")
st.write("Enter a city and get clothing + outdoor activity advice based on today's weather.")


# --- Part A: weather function ---
def get_current_weather(location):
    """Fetch current + today's forecast weather from wttr.in for a given location."""
    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f"wttr.in error: status {response.status_code}")
    try:
        data = response.json()
    except ValueError:
        raise Exception(f"Could not find a location named {location}")

    current = data["current_condition"][0]
    today = data["weather"][0]

    # Pull a few hourly points so advice can reflect change through the day
    hourly_snapshot = [
        {
            "time": f"{int(h['time']) // 100:02d}:00",
            "tempF": h["tempF"],
            "chanceofrain": h["chanceofrain"],
        }
        for h in today["hourly"][::3]  # every ~9 hours: morning/afternoon/evening
    ]

    return {
        "location": location,
        "temperature_F": float(current["temp_F"]),
        "feels_like_F": float(current["FeelsLikeF"]),
        "description": current["weatherDesc"][0]["value"],
        "humidity_pct": current["humidity"],
        "wind_mph": current["windspeedMiles"],
        "chance_of_rain_pct": current.get("chanceofrain", "0"),
        "today_min_F": today["mintempF"],
        "today_max_F": today["maxtempF"],
        "hourly_snapshot": hourly_snapshot,
    }


# --- Part B: tool definition for OpenAI ---
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get current and today's weather for a city, zip code, or landmark.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City, zip code, airport code, or landmark. "
                                    f"Default to '{DEFAULT_LOCATION}' if user gives no location.",
                }
            },
            "required": ["location"],
        },
    },
}

city_input = st.text_input("City (leave blank to use Syracuse, NY):", "")

if st.button("Get advice"):
    location = city_input.strip() if city_input.strip() else DEFAULT_LOCATION

    messages = [
        {"role": "system", "content": "You are a helpful weather-based outfit and activity advisor."},
        {"role": "user", "content": f"What should I wear today in {location}, and what outdoor "
                                     f"activities would suit the weather there?"},
    ]

    # --- First call: let the model decide to invoke the tool ---
    first_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[weather_tool],
        tool_choice="auto",
    )
    response_message = first_response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)  # assistant's tool-call turn

        for tool_call in tool_calls:
            args = json.loads(tool_call.function.arguments)
            loc = args.get("location", DEFAULT_LOCATION)

            try:
                weather_data = get_current_weather(loc)
            except Exception as e:
                weather_data = {"error": str(e)}

            with st.expander("Weather data used"):
                st.json(weather_data)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(weather_data),
            })

        # --- Second call: model writes the actual advice using weather data ---
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        st.write(second_response.choices[0].message.content)
    else:
        # Model answered without needing the tool (shouldn't normally happen here)
        st.write(response_message.content)