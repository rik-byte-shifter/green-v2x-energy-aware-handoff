from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal


@dataclass(frozen=True)
class WeatherProfile:
    """Weather conditions affecting wireless propagation.

    Values are expressed in dB / dB/km style to directly integrate into the
    simplified path-loss + shadowing model used by the simulator.
    """

    name: str
    path_loss_exponent: float
    shadowing_std_db: float
    rain_attenuation_db_per_km: float
    description: str


# Standard profiles for V2X papers.
# Note: `rain_attenuation_db_per_km` is also used as a generic "extra loss"
# term for precipitation / scattering conditions (snow, fog, storms, dust).
WEATHER_PROFILES: Dict[str, WeatherProfile] = {
    "clear": WeatherProfile(
        name="Clear/Sunny",
        path_loss_exponent=3.0,
        shadowing_std_db=6.0,
        rain_attenuation_db_per_km=0.0,
        description="Ideal conditions, minimal attenuation.",
    ),
    "light_rain": WeatherProfile(
        name="Light Rain",
        path_loss_exponent=3.5,
        shadowing_std_db=9.0,
        rain_attenuation_db_per_km=1.5,
        description="Light precipitation, moderate attenuation.",
    ),
    "heavy_rain": WeatherProfile(
        name="Heavy Rain",
        path_loss_exponent=4.0,
        shadowing_std_db=11.0,
        rain_attenuation_db_per_km=7.0,
        description="Heavy precipitation, significant attenuation.",
    ),
    "snow": WeatherProfile(
        name="Snow",
        path_loss_exponent=4.2,
        shadowing_std_db=13.0,
        rain_attenuation_db_per_km=5.0,
        description="Snow accumulation and scattering.",
    ),
    "fog": WeatherProfile(
        name="Fog/Mist",
        path_loss_exponent=3.7,
        shadowing_std_db=12.0,
        rain_attenuation_db_per_km=2.0,
        description="Reduced visibility and scattering effects.",
    ),
    "thunderstorm": WeatherProfile(
        name="Thunderstorm",
        path_loss_exponent=4.5,
        shadowing_std_db=16.0,
        rain_attenuation_db_per_km=15.0,
        description="Severe weather, maximum attenuation (EMI effects simplified into loss).",
    ),
    "dust_sand_storm": WeatherProfile(
        name="Dust/Sand Storm",
        path_loss_exponent=4.2,
        shadowing_std_db=14.0,
        rain_attenuation_db_per_km=4.0,
        description="Particle scattering and extra path loss.",
    ),
}


WeatherType = Literal[
    "clear",
    "light_rain",
    "heavy_rain",
    "snow",
    "fog",
    "thunderstorm",
    "dust_sand_storm",
]


def get_weather_profile(weather_profile: WeatherType | str) -> WeatherProfile:
    """Resolve a weather key into a profile, defaulting to `clear`."""

    if isinstance(weather_profile, str) and weather_profile in WEATHER_PROFILES:
        return WEATHER_PROFILES[weather_profile]
    # If caller passes an unknown string, use clear rather than throwing.
    return WEATHER_PROFILES["clear"]

