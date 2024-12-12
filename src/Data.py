import datetime
import requests
import os


def download_tle():
    """
    Reads satellite two-line elements (TLE) from celestrack and saves them as a local text file

    :return: file_string: TLE file name containing the current date in UTC time
    """
    now = datetime.datetime.utcnow()
    file_string = "outputs/" + "TLE " + now.strftime("%m-%d-%Y") + ".txt"

    #check if file exists, if so, return file name
    if os.path.isfile(file_string):
        return file_string

    # url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
    url = 'https://celestrak.com/NORAD/elements/visual.txt'
    r = requests.get(url, allow_redirects=True, timeout=10)
    open(file_string, 'wb').write(r.content)
    return file_string
