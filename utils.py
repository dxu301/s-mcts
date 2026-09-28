import json
import os
import re
import sys

def read_jsonl_to_list(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data

def duration_to_hhmm(duration_str):
    """
    Convert strings like '3 hours 11 mins', '1 hour 50 min', or '50 mins' to HH:MM format.
    Handles both singular and plural forms of 'hour' and 'min'.
    """

    if(duration_str==None):
        return duration_str

    hours = 0
    minutes = 0
    
    # Extract hours (handles "hour" or "hours")
    hour_match = re.search(r'(\d+)\s*hour[s]?', duration_str)
    if hour_match:
        hours = int(hour_match.group(1))
    
    # Extract minutes (handles "min" or "mins")
    minute_match = re.search(r'(\d+)\s*min[s]?', duration_str)
    if minute_match:
        minutes = int(minute_match.group(1))
    
    return f"{hours:02}:{minutes:02}"


def parse_travel_detail(detail):
    detail = detail.strip()

    # Check for "no flight" case
    no_flight_pattern = r"There is no flight from (.+?) to (.+?) on (\d{4}-\d{2}-\d{2})"
    match = re.match(no_flight_pattern, detail)
    if match:
        return {
            "name": "Flight",
            "mode": "flight",
            "from_to": f"{match.group(1)}_{match.group(2)}",
            # "date": match.group(3),
            "duration": None,
            "distance_km": None,
            "cost": None
        }

    # Generic parsing for taxi/self-driving
    generic_pattern = r"([\w-]+), from (.+?) to (.+?), duration: (.*?), distance: ([\d.,]+) km, cost: (\d+)"
    match = re.match(generic_pattern, detail)
    if match:
        return {
            "mode": match.group(1),
            "name": match.group(1).capitalize(),
            "from_to": f"{match.group(2)}_{match.group(3)}",
            # "date": None,
            "duration": duration_to_hhmm(match.group(4)),
            "distance_km": float(match.group(5).replace(",", "")),
            "cost": float(match.group(6))
        }

    return {"raw": detail}  # fallback if no pattern matches


def restructure_data(flat_data):
    hierarchical = {}
    travel = []

    for key, value in flat_data.items():
        # Keys with ' in ' (e.g., "Restaurants in Toledo")
        if " in " in key:
            cat_part, location = key.split(" in ", 1)
            category = cat_part.strip().lower()

            if(category=="restaurants"):
                for rest in value:
                    rest["Cuisines"] = rest["Cuisines"].split(",")
                    rest["Cuisines"] = [a.strip() for a in rest["Cuisines"]]
                    rest["name"] = rest["Name"]
                    rest["Average Cost Per Person"] = rest["Average Cost"]
                    del rest["Average Cost"]
                    del rest["Name"]
                    del rest["Aggregate Rating"]

                
            if(category=="attractions"):
                for att in value:
                    att["name"] = att["Name"]
                    del att["Name"]
                    del att["Latitude"]
                    del att["Longitude"]
                    del att["Address"]
                    del att["Phone"]
                    del att["Website"]

            if(category=="accommodations"):
                for accom in value:
                    accom["house_rules"] = accom["house_rules"].split("&")
                    accom["house_rules"] = [a.strip() for a in accom["house_rules"]]
                    accom["name"] = accom["NAME"]
                    accom["City"] = accom["city"] 
                    accom["price_per_room"] = accom["price"]
                    del accom["price"]
                    del accom["NAME"]
                    del accom["city"]
                    del accom["review rate number"]
                
            if(category not in hierarchical):
                hierarchical[category] = value
            else:
                hierarchical[category].extend(value)


        else:
            # Travel keys: "Taxi from X to Y", "Flight from X to Y on DATE"
            m = re.match(r"(Flight|Taxi|Self-driving) from (.+?) to (.+?)(?: on (\d{4}-\d{2}-\d{2}))?$", key)
            if m:
                if(type(value)==str):
                    d = parse_travel_detail(value)

                    if("raw" in d):
                        continue

                    if("cost" in d and d["cost"]==None):
                        continue
                    
                    del d["duration"]
                    del d["distance_km"]
                    travel.append(d)
                else:
                    for v in value:
                        v["mode"] = "flight"
                        v["cost"] = v["Price"]
                        if(v["cost"]==None):
                            continue

                        origin = v["OriginCityName"]
                        dest = v["DestCityName"]
                        v["from_to"] = f"{origin}_{dest}"
                        v["duration"] = duration_to_hhmm(v["ActualElapsedTime"])
                        # v["date"] = v["FlightDate"]
                        # v["distance_km"] = v["Distance"] 
                        v["Flight DepTime"] = v["DepTime"]
                        v["Flight ArrTime"] = v["ArrTime"]
                        flight_number = v["Flight Number"]
                        v["name"] = f"Flight, {flight_number}"

                        del v["Price"]
                        del v["OriginCityName"]
                        del v["DestCityName"]
                        del v["ActualElapsedTime"]
                        del v["FlightDate"]
                        del v["Distance"]
                        del v["DepTime"]
                        del v["ArrTime"]
                        del v["Flight Number"]
                        del v["Flight DepTime"]
                        del v["Flight ArrTime"]
                        del v["duration"]
                        travel.append(v)


    # Attach travel
    hierarchical["transport"] = travel


    return hierarchical




