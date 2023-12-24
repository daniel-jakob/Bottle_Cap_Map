import requests

def get_geocoordinates(address):
    """
    Get the geographic coordinates of an address from OpenStreetMap's Nominatim service.

    Parameters:
    address (str): The address to geocode.

    Returns:
    tuple: The latitude and longitude of the address, or None if the geocoding request failed.
    """
    response = requests.get(
        'https://nominatim.openstreetmap.org/search',
        params={'q': address, 'format': 'json'}
    )

    data = response.json()

    if data:
        return float(data[0]['lon']), float(data[0]['lat'])

    return None