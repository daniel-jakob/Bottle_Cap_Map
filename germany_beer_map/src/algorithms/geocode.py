import requests
import numpy as np

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

def convert_address_to_coords(csvfile):
    # Read in bottle_caps.csv
	bottle_caps = np.genfromtxt(csvfile, delimiter=",", dtype=str, skip_header=1)

	bottle_cap_coords = []

	# bottle_cap_coords = [(13.0944453, 54.2907393), (11.591350844509808, 53.37741685), (12.459549961499999, 50.52140085), (7.956500987110655, 50.9910225), (9.653526, 51.4176975), (11.553636633022338, 48.14608575), (11.9069408, 48.3068168), (6.654346047066047, 51.83582915), (10.8870087, 49.889238199999994), (10.227481824523348, 50.45433915), (11.606263141955619, 50.1081385), (7.1418535749854755, 51.26109235)]

	# Get the geocoordinates of each bottle cap
	for i in range(len(bottle_caps)):
		bottle_cap = get_geocoordinates(bottle_caps[i][2])
		bottle_cap_coords.append(bottle_cap)

	return bottle_cap_coords
