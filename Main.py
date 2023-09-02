# pylint: disable=line-too-long
import Data
import datetime
from geopy.geocoders import Nominatim
import os
from pathlib import Path
from Site import Site
import sys
from astropy.time import Time
from astropy.coordinates import SkyCoord
from skyfield.api import load, wgs84
from skyfield import almanac
import pandas as pd
from PyAstronomy import pyasl
from tqdm import tqdm



start_time = datetime.datetime.utcnow()
stop_time = start_time + datetime.timedelta(days=1)
site_name = "Beijing"
# polaris coordinates(RA, Dec)
star_coord = (2.530301, 89.264109)

def main():
    """
    Reads satellite elements from TLE, takes user input, then propagates all satellites and saves all observations.

    `Steps`

    1. Downloads two-line element (TLE) file and read elements into a list of Satellite objects
    2. Sets observation time frame
        By default, time begins when the program is run and ends 24 hours later
    3. Instantiates observation site.
        The user inputs a general location and geopy matches an exact geocode location
        The user has the option to name the site so that it appears in the .csv file output, but this optional
    4. Propagates satellites
        The propagator predicts the orbit of each satellite in the Sat list and finds opportunities where the satellite
        is visible from the observation location after sunset and before sunrise.
    5. Saves observations
        All predicted opportunities are saved to a sqlite3 database and a .csv file
    """
    print("Building directories and downloading TLEs...")
    # Confirming the `outputs` directory exists
    cwd_path = os.getcwd()
    Path(cwd_path + '/outputs').mkdir(parents=False, exist_ok=True)

    # Downloads satellite elements from celestrack, gets file name in return
    tle_file = Data.download_tle()

    # Load satellite two-line elements from a file.
    satellites = load.tle_file(tle_file)
    print(f'Loaded {len(satellites)} satellites')
    
    # Instantiates a Site object, gets longitude, latitude, altitude of the site
    new_site = find_site(site_name)

    print()
    print(new_site.obs_location)
    print("Lon: {}\nLat: {}\nElv: {}m\nTZ: {}\n".format(
        new_site.lon, new_site.lat, new_site.altitude, new_site.timezone))

    # Check if any satellite is visible at night at the observation site between start and stop time
    check_if_visible(satellites, new_site)

    # Check if any satellite cross a coordinate(RA, Dec) at night at the observation site between start and stop time
    coord = SkyCoord(star_coord[0], star_coord[1], unit='deg')
    check_if_cross(satellites, coord, new_site)


def find_site(site_name):
    try:
        geolocator = Nominatim(user_agent="Meteor Detection")
        location = geolocator.geocode(site_name)
        new_site = Site(location)  # Instantiating Site object
    except AttributeError:
        print(f"{site_name} didn't match to a longitude or latitude. "
              "Try entering the location again with a different representation\n "
              "(Tip: check spelling, or try a zip code.)")
        sys.exit(1)

    return new_site


def check_if_visible(satellites, obs_site):
    local_date = (start_time + obs_site.utc_offset).strftime("%m-%d-%Y")
    obs_file = "outputs/OBS_{}".format(local_date)

    # Create a timescale object
    ts = load.timescale()
    
    # set the start and stop time
    t0 = ts.from_astropy(Time(start_time, scale='utc'))
    t1 = ts.from_astropy(Time(stop_time, scale='utc'))
    visible_satellites = []
    for satellite in tqdm(satellites):
        visible_state = 'not visible'
        # Create a Topos object representing the observer location
        site = wgs84.latlon(obs_site.lat, obs_site.lon, elevation_m=obs_site.altitude)
        
        # Find the next three culminations of the satellite between start and stop time
        t, events = satellite.find_events(site, t0, t1, altitude_degrees=30.0)

        # Check if the satellite is visible at night at the observation site between start and stop time
        eph = load('de421.bsp')
        sunlit = satellite.at(t).is_sunlit(eph)
        f = almanac.sunrise_sunset(eph, site)

        for ti, sunlit_flag in zip(t, sunlit):
            sunup_flag = f(ti)
            if sunlit_flag and not sunup_flag:
                visible_state = 'visible'
                break
        
        if visible_state == 'visible':
            visible_satellites.append(satellite.name)

    print(f"There are {len(visible_satellites)} satellites visible in the night sky of {site_name} between {start_time} and {stop_time}")
    pd.DataFrame(visible_satellites, columns=['NAME']).to_csv(obs_file, index=False)


def check_if_cross(satellites, coord, obs_site):
    local_date = (start_time + obs_site.utc_offset).strftime("%m-%d-%Y")
    obs_file = "outputs/CRO_{}".format(local_date)

    site = wgs84.latlon(obs_site.lat, obs_site.lon, elevation_m=obs_site.altitude)

     # Create a timescale object
    ts = load.timescale()
    
    # set the start and stop time
    t0 = ts.from_astropy(Time(start_time, scale='utc'))
    t1 = ts.from_astropy(Time(stop_time, scale='utc'))

    crossed_satellites = []
    for satellite in tqdm(satellites):
        working_time = t0

        while working_time.tt < t1.tt:
            geocentric = satellite.at(working_time)
            difference = satellite - site
            topocentric = difference.at(working_time)
            ra, dec, distance = topocentric.radec(epoch=satellite.epoch)

            # Calculate angular distance (in DEGREES) between satellite and the calibration stellar coordinate
            ang_sep = pyasl.getAngDist(
                ra._degrees, dec._degrees, coord.ra.value, coord.dec.value)
            if ang_sep <= 0.5:
                print(f"{satellite.name} crosses the calibration star at {working_time.utc_datetime()}")
                crossed_satellites.append(satellite.name)
                break

            working_time = working_time + datetime.timedelta(seconds=100)

    print(f"There are {len(crossed_satellites)} satellites cross the calibration star between {start_time} and {stop_time}")
    pd.DataFrame(crossed_satellites, columns=['NAME']).to_csv(obs_file, index=False)

if __name__ == "__main__":
    main()
