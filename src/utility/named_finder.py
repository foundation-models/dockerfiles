import json
import re
import pandas as pd

"""
TO BE NOTED:
    Output Example: 
    {"country": [United States, (s,e), US, (s,e), USA, (s,e)], "state": [Florida, (s,e), FL, (s,e)], "city": [Tampa, (s,e)]}    

    1) The input address must be filtered for extra characters using: 
    address = ''.join(e for e in address if (e.isalpha() or e.isspace())).strip()

    Otherwise, the (start, end) positions returned by the code wont be accurate.

    2) In the case of having a country with a similar 'state' and 'city' name, both appear in output,
    for example, state: Kansas, city: Kansas

    3) If Multiple Country data found in address: 
    Output format: [{Country 1 Entities}, {Country 2 Entities}, ...]
"""

def checkString(key, value, address, output):
    ''' compares a string value of json with input address '''
    if type(value) != str:
        return
    if re.search(r"\b" + re.escape(value) + r"\b", address):
        prepareOutput(key, value, output)
    return output

def prepareOutput(key, value, output):
    if not(len(value)):
        return
    
    if key in output:
        if value not in output[key]:
            output[key].append(value)
    else:
        output[key] = [value]
    return output

def checkLists(l1, l2):
    for item in l1:
        if item in l2: 
            return True   
    return False

class NamedFinder():
    def __init__(self, address):
        self.address = address
        
    def load_data(self, data_file):
        with open(data_file, 'r', encoding="utf-8") as f:
            data = json.loads(f.read())
        return data
        
    def getCity(self, country_data):
        ''' Given the input address, find all cities that match countries json data '''
        country = country_data['country']
        
        output = {}
        for state in country['states']:
            for city in state['cities']:
                output = checkString('city', city, self.address, output)
        
        output = self.getState(country, output)
        country_fields = self.getCountry(country, output)
        verified_fields = verifyCity(country_fields, self.address)
        return verified_fields

    def getState(self, country, output):
        ''' Given identified cities from address, find states, 
            then verify if states in address, else remove cities '''
        
        if 'city' in output: 
            for city in output['city']:
                if not isinstance(city, str):
                    continue
                for state in country['states']:
                    if city in state['cities']:
                        output = checkString('state', state['name'], self.address, output)
                        for item in state['aliases']:
                            output = checkString('state', item, self.address, output)
                        
            if 'state' in output:
                verified = verifyState(country['states'], output['state'], output['city'])
            else: 
                return output
            output['city'] = verified
        
        else: # No city found, check if State in string
            for state in country['states']:
                output = checkString('state', state['name'], self.address, output)
                for item in state['aliases']:
                    output = checkString('state', item, self.address, output)
        return output
    
    def getCountry(self, country, output):
        ''' Given identified states, find and verify their countries '''
        
        if 'state' not in output:
            return output
        
        for state in country['states']:
            if checkLists(output['state'], [state['name']]+state['aliases']):
                for item in [country['name']]+country['aliases']:
                    output = checkString('country', item, self.address, output)
        return output

def verifyState(country, states, cities):
    ''' Check if elements of output address belong to the same branch, 
        i.e., the same country'''
    verified =  [] 

    for state in country:
        if checkLists(states, state['aliases']+[state['name']]):
            for city in cities:
                if city in state['cities']:  
                    verified.append(city)
    return verified

def check_contained(ls):
    contained = []
    for l1 in ls:
        for l2 in ls:
            if l1 != l2:
                if l1 in l2:
                    contained.append(l1)
    return contained

def verifyCity(fields, address):
    ''' Check identified city not a street address in the same state'''
    
    if 'city' not in fields:
        return fields

    ''' Rule to exclude a contained identified city:
        if a city is part of another city and both are in the address (South San Francisco, San Francisco)'''
    contained = check_contained(fields['city'])
    if len(contained):
        for city in contained:
            if city in fields['city']:
                fields['city'].remove(city)

    ''' Rules to exclude an identified city:
        Rule 1: if a number is immediately before or after the city (345 Park Ave),
        Rule 2: if an ordinal number is immediately before or after the city (19th street),
        Rule 3: if a single (alphabetic) character is immediately before or after the city (N, S, E, W Street), 
        '''
    rules = [r'\b(\d+)\b\s+\b{}\b|\b{}\b\s+\b(\d+)\b', 
            r'\b(\d+(?:st|nd|rd|th))\b\s+\b{}\b|\b{}\b\s+\b(\d+(?:st|nd|rd|th))\b',
            r'\b[a-zA-Z]\b\s+\b{}\b|\b{}\b\s+\b[a-zA-Z]\b']
    
    for rule in rules:
        for city in fields['city']:
            regex = re.compile(rule.format(city, city))
            match = bool(regex.search(address))
            if match and city in fields['city']:
                fields['city'].remove(city)

    return fields

def getLocations(fields, address):
    if not fields:
        return 

    if not len(fields.keys()):
        return
    
    for key in fields:
        for i in range(0, len(fields[key]),2):
            location = re.search(r"\b" + re.escape(fields[key][i]) + r"\b", address).span()
            fields[key].append(location)

    return fields

def extract_locations(data, address):
    """ extracts the city, state, and country from the given address

    Args:
        data (json): countries, states, and cities
        address (string): the address which the geographies are needed to extract from.

    Returns:
        list: list of dictionary (city, state, country)
    """
    findNames = NamedFinder(address)

    # iterate in json data to extract address fields: row is the country data
    results = []
    for i, row in enumerate(data):
        address_fields = findNames.getCity(row)
        results.append(getLocations(address_fields, address))
        
    results = list(filter(lambda x: x is not None, results))
    
    """ If Multiple Country data found:
        keep the country with more number of identified entities,
        if both have the same number of entities, keep both. """
    if len(results) > 1:
        lens = (lambda x:[len(i) for i in x])(results)
        if min(lens) < max(lens):
            results = max(results, key=len)
    return results

def main():
    # Test Input address
    addr =  "CA, US"
    addr1 = '848 avenu Atlanta, GA 30308, US'
    addr2 = """All the Best,    Alison Thomas
    Your Bay Area Expert Realtor®
    CalDRE#: 02145674
    225 N Santa Cruz Ave
    Los Gatos, CA 95030
    m: 408.835.2027"""
    addr3 = """
    HackerX
    MOLOCO 601 Marshall Street, Redwood City, CA 94063 US
    Unsubscribe  |  Privacy Policy"""
    addr4 = """
    Kris Young
    Director
    BHO Tech 
    San Jose, CA, USA
    
    H. Akbari
    Senrior ML Engineer
    Intapp
    Fredericton, NB, Canada
    """
    addr5 = """
    Jerico Delarouse
    Associate Financial Consultant | , CA
    Phone: 650-313-3059
    Los Gatos Branch
    240 Third Street, Suite 100
    Los Gatos, CA. 94022"
    """
    
    addr6="""
    Sincerely,

    Lennox Donlon
    Financial Planner
    CA License #4073322
    
    One Birch Park Drive, 16th Floor  |  Boston, MA 02210
    """
    # load the data into a json structure.
    json_path = "../conf/countries.json" # JSON File
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())


    
if __name__ == "__main__":
    main()