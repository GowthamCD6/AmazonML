"""
Realistic Dataset Generator for Amazon ML Challenge 2026 — Business Entity Resolution
Generates realistic multi-source business entity data matching the exact competition schema and noise distributions.
"""

import os
import random
import re
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

COUNTRIES = ["US", "GB", "IN", "CA", "AU", "DE", "SG", "FR", "JP", "NL"]

LEGAL_SUFFIX_MAP = {
    "US": ["LLC", "Inc.", "Corp.", "Corporation", "Incorporated", "Co.", "LLP"],
    "GB": ["Ltd.", "Limited", "PLC", "LLP", "Co. Ltd."],
    "IN": ["Pvt. Ltd.", "Private Limited", "Pvt Ltd", "Ltd.", "Limited", "LLP", "Enterprises"],
    "CA": ["Inc.", "Corp.", "Ltée", "Limited", "ULC"],
    "AU": ["Pty Ltd", "Proprietary Limited", "Ltd", "NL"],
    "DE": ["GmbH", "AG", "GmbH & Co. KG", "UG"],
    "SG": ["Pte. Ltd.", "Private Limited", "Ltd.", "LLP"],
    "FR": ["SAS", "SARL", "SA", "SNC"],
    "JP": ["K.K.", "Kabushiki Kaisha", "G.K."],
    "NL": ["B.V.", "N.V."]
}

CORE_NAMES = [
    "Acme Global", "Apex Logistics", "Beacon Health", "BlueSky Technologies",
    "BrightPath Financial", "Cascade Capital", "Catalyst Innovation", "Cedar Point Energy",
    "Centurion Security", "Cinnabar Creative", "Clearwater Analytics", "CloudScale Systems",
    "Cobalt Media Group", "Compass Point Marine", "CoreLogic Solutions", "Crestview Holdings",
    "CrossRoads Transport", "CyberShield Defense", "Delta Wave Electronics", "Dynamo Retail",
    "Eagle Eye Surveillance", "EcoSphere Solutions", "Elevate Digital", "Emerald City Foods",
    "Enclave Networks", "Envision Healthcare", "Equinox Venture Partners", "Evergreen Forestry",
    "Falcon Heavy Transport", "First Horizon Banking", "Fleetwood Engineering", "Frontier Biotech",
    "Fusion Power Dynamics", "Galaxy Softworks", "Gateway Telecom", "Genesis Industrial",
    "Global Trade Alliance", "Golden Gate Trading", "Grandview Real Estate", "GreenLeaf Organics",
    "Guardian Asset Management", "Halo Interactive", "Harbor Light Logistics", "Harmonic Audio",
    "Helix BioSciences", "Heritage Distilling", "Horizon Freight", "Hyperion Solar",
    "Iconic Architecture", "Infinity Robotics", "Insightful Analytics", "Integrity Consulting",
    "Ion Propulsion Labs", "Ironclad Construction", "Ivy League Education", "Jade Palace Dining",
    "Jupiter Aerospace", "Keystone Civil Works", "Kinetic Sports", "Kodiak Tactical",
    "Laguna Marine", "Lantern Publishing", "Latitude Geospatial", "Legacy Manufacturing",
    "Liberty Title Agency", "Lighthouse Hospitality", "Linear Motion Control", "Luminary Lighting",
    "Magnolia Hospitality", "Matrix Computing", "Maven Data Science", "Meridian Wealth",
    "Metro Urban Transit", "MicroTech Semiconductor", "Milestone Education", "Modern Precision",
    "Monarch Aerospace", "Nautilus Oceanic", "Navitas Clean Energy", "Nebula Cloud Hosting",
    "New Horizon Health", "Nexus Telecom", "Nimble Supply Chain", "NorthStar Financial",
    "Nova Medical Devices", "Oakwood Furnishings", "Oceanic Seafoods", "Olympus Chemical",
    "OmniSource Global", "OnPoint Diagnostics", "Optima Insurance", "Orbit Satellite Services",
    "Overland Express", "Pacific Rim Trading", "Paladin Risk Management", "Paramount Construction",
    "Pathfinder Robotics", "Peak Performance Fitness", "Pinnacle Pharmaceuticals", "Pioneer Drilling",
    "Polaris Navigation", "Prairie Wind Renewables", "Precision Machining", "Premier Dental Care",
    "Prime Auto Group", "Prism Optical", "ProActive Safety", "Prosper Wealth Advisors",
    "Pulse Digital Marketing", "Quantum Computing Labs", "Quest Diagnostics Global", "Radiant Energy",
    "Radius Geospatial", "Redwood Software", "Reliant Power Systems", "Renaissance Arts",
    "Resolute Mining", "Rigel Marine Services", "Riverstone Capital", "Rockwell Automation Systems",
    "Safeguard Insurance", "Sanctuary Spa Group", "Sapphire Healthcare", "Saturn Heavy Industries",
    "Seaside Seafood Co", "Sentient AI Systems", "Serenity Living", "Sierra Nevada Brewing",
    "Sigma BioPharma", "Silicon Valley Micro", "Silverline Transport", "Skyline Real Estate",
    "Solaria Energy Systems", "SoundWave Media", "Southland Agriculture", "Spectra Chemical",
    "Spectrum Communications", "Spire Global Solutions", "Standard Petroleum", "Starfire Gaming",
    "Starlight Entertainment", "Sterling Bank & Trust", "Summit Mountain Gear", "Sunlight Solar Power",
    "Sunrise Bakery", "Supreme Logistics", "Synergy Pharmaceuticals", "Synthesis Data Labs",
    "Talon Security", "Tandem Software", "Taylor & Associates", "TerraFirma Landscaping",
    "Threshold Interactive", "Timberline Lumber", "Titan Heavy Machinery", "TopTier Athletics",
    "TransNational Logistics", "Treasure Coast Marine", "Trident Subsea Works", "TriState Utilities",
    "TrueNorth Consulting", "Turbine Power Systems", "Ultratech Industrial", "Unified Communications",
    "United Parcel Express", "Universal Aerospace", "Vanguard Asset Management", "Velocity Motors",
    "Venture Point Capital", "Veritas Legal Services", "Vertex Engineering", "Vibrant Media Group",
    "Victory Marine", "Viking Freight", "Visionary Healthcare", "Vista Window Systems",
    "Volt Electrical Services", "Voyager Travel Group", "Waveform Acoustic", "Wayfarer Outdoors",
    "Wellspring Health", "West Coast Distribution", "Western Digital Systems", "Whispering Pines Resort",
    "White Cloud Technologies", "Wildcat Oil Exploration", "Windward Nautical", "WorldBridge Express",
    "Xcel Precision Tools", "Zenith Broadcast Group", "Zephyr Clean Energy", "Zion Natural Foods"
]

STREETS = [
    "Main St", "High Street", "Park Avenue", "Broadway", "Market Street", "Industrial Boulevard",
    "MG Road", "Brigade Road", "Oxford Street", "Queen Street", "Station Road", "Commercial Road",
    "Bay Street", "George Street", "Friedrichstrasse", "Rue de Rivoli", "Orchard Road", "Church St",
    "Elm Street", "Maple Ave", "Washington Blvd", "King Street", "Victoria Road", "Harbor Drive"
]

CITIES = {
    "US": ["New York", "San Francisco", "Austin", "Chicago", "Seattle", "Boston", "Atlanta", "Dallas"],
    "GB": ["London", "Manchester", "Birmingham", "Leeds", "Glasgow", "Bristol", "Edinburgh"],
    "IN": ["Bengaluru", "Mumbai", "New Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata"],
    "CA": ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa"],
    "AU": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"],
    "DE": ["Berlin", "Munich", "Frankfurt", "Hamburg", "Stuttgart"],
    "SG": ["Singapore"],
    "FR": ["Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux"],
    "JP": ["Tokyo", "Osaka", "Nagoya", "Fukuoka", "Yokohama"],
    "NL": ["Amsterdam", "Rotterdam", "Utrecht", "The Hague"]
}

def perturb_name(name, country):
    choice = random.random()
    suffixes = LEGAL_SUFFIX_MAP.get(country, ["Inc.", "Ltd."])
    
    # 1. Legal suffix mutation
    if choice < 0.25:
        suf = random.choice(suffixes)
        return f"{name} {suf}"
    elif choice < 0.40:
        # Abbreviation
        name = name.replace("Technologies", "Tech").replace("Solutions", "Solns").replace("International", "Intl")
        name = name.replace("Systems", "Sys").replace("Corporation", "Corp").replace("Enterprises", "Ent")
        return name
    elif choice < 0.55:
        # Punctuation / case / spacing
        if random.random() < 0.5:
            return name.replace(" ", "-").lower()
        else:
            return name.upper()
    elif choice < 0.70:
        # Typo
        if len(name) > 6:
            idx = random.randint(1, len(name) - 2)
            c = name[idx]
            name = name[:idx] + name[idx+1:] if random.random() < 0.5 else name[:idx] + c + name[idx:]
        return name
    elif choice < 0.85:
        # Word order / DBA name
        parts = name.split()
        if len(parts) >= 2:
            return f"{parts[-1]} {parts[0]}"
        return name
    else:
        return name

def perturb_address(street_num, street, city, country, postal):
    choice = random.random()
    if choice < 0.20:
        st = street.replace("Street", "St.").replace("Avenue", "Ave").replace("Boulevard", "Blvd").replace("Road", "Rd")
        return f"{street_num} {st}, {city}, {postal}, {country}"
    elif choice < 0.40:
        return f"{street_num} {street}, {city}"
    elif choice < 0.60:
        suite = f"Suite {random.randint(100, 999)}"
        return f"{suite}, {street_num} {street}, {city} - {postal}"
    elif choice < 0.80:
        return f"{city}, {street_num} {street}, {postal}"
    else:
        return f"{street_num} {street}, {city}, {postal}"

def generate_dataset(num_train_s1=2500, num_test_s1=1000):
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    
    for split, num_s1 in [("train", num_train_s1), ("test", num_test_s1)]:
        s1_records = []
        s2_records = []
        s3_records = []
        ground_truth = []
        
        s2_counter = 1
        s3_counter = 1
        
        for i in range(1, num_s1 + 1):
            s1_id = f"S1_{split.upper()}_{i:06d}"
            base_core = random.choice(CORE_NAMES) + f" {random.randint(1, 9999)}"
            country = random.choice(COUNTRIES)
            city = random.choice(CITIES[country])
            street = random.choice(STREETS)
            street_num = random.randint(1, 9999)
            postal = f"{random.randint(10000, 99999)}"
            
            s1_name = f"{base_core} {random.choice(LEGAL_SUFFIX_MAP[country])}"
            s1_addr = f"{street_num} {street}, {city}, {postal}, {country}"
            s1_records.append({
                "entity_id": s1_id,
                "business_name": s1_name,
                "business_address": s1_addr,
                "country": country
            })
            
            match_prob = random.random()
            matched_ids = []
            
            if match_prob < 0.20:
                pass
            elif match_prob < 0.65:
                if random.random() < 0.5:
                    s2_id = f"S2_{split.upper()}_{s2_counter:06d}"
                    s2_counter += 1
                    s2_name = perturb_name(base_core, country)
                    s2_addr = perturb_address(street_num, street, city, country, postal)
                    s2_country = country if random.random() > 0.1 else ""
                    s2_records.append({"entity_id": s2_id, "business_name": s2_name, "business_address": s2_addr, "country": s2_country})
                    matched_ids.append(s2_id)
                else:
                    s3_id = f"S3_{split.upper()}_{s3_counter:06d}"
                    s3_counter += 1
                    s3_name = perturb_name(base_core, country)
                    s3_addr = perturb_address(street_num, street, city, country, postal)
                    s3_country = country if random.random() > 0.1 else ""
                    s3_records.append({"entity_id": s3_id, "business_name": s3_name, "business_address": s3_addr, "country": s3_country})
                    matched_ids.append(s3_id)
            elif match_prob < 0.90:
                s2_id = f"S2_{split.upper()}_{s2_counter:06d}"
                s2_counter += 1
                s2_name = perturb_name(base_core, country)
                s2_addr = perturb_address(street_num, street, city, country, postal)
                s2_country = country if random.random() > 0.1 else ""
                s2_records.append({"entity_id": s2_id, "business_name": s2_name, "business_address": s2_addr, "country": s2_country})
                matched_ids.append(s2_id)
                
                s3_id = f"S3_{split.upper()}_{s3_counter:06d}"
                s3_counter += 1
                s3_name = perturb_name(base_core, country)
                s3_addr = perturb_address(street_num, street, city, country, postal)
                s3_country = country if random.random() > 0.1 else ""
                s3_records.append({"entity_id": s3_id, "business_name": s3_name, "business_address": s3_addr, "country": s3_country})
                matched_ids.append(s3_id)
            else:
                for _ in range(2):
                    s2_id = f"S2_{split.upper()}_{s2_counter:06d}"
                    s2_counter += 1
                    s2_name = perturb_name(base_core, country)
                    s2_addr = perturb_address(street_num, street, city, country, postal)
                    s2_country = country if random.random() > 0.1 else ""
                    s2_records.append({"entity_id": s2_id, "business_name": s2_name, "business_address": s2_addr, "country": s2_country})
                    matched_ids.append(s2_id)
                s3_id = f"S3_{split.upper()}_{s3_counter:06d}"
                s3_counter += 1
                s3_name = perturb_name(base_core, country)
                s3_addr = perturb_address(street_num, street, city, country, postal)
                s3_country = country if random.random() > 0.1 else ""
                s3_records.append({"entity_id": s3_id, "business_name": s3_name, "business_address": s3_addr, "country": s3_country})
                matched_ids.append(s3_id)
                
            ground_truth.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(matched_ids)
            })

        for _ in range(int(num_s1 * 0.3)):
            s2_id = f"S2_{split.upper()}_{s2_counter:06d}"
            s2_counter += 1
            b_name = random.choice(CORE_NAMES) + f" Distractor {random.randint(100, 999)}"
            c = random.choice(COUNTRIES)
            city = random.choice(CITIES[c])
            st = random.choice(STREETS)
            s2_records.append({
                "entity_id": s2_id,
                "business_name": f"{b_name} {random.choice(LEGAL_SUFFIX_MAP[c])}",
                "business_address": f"{random.randint(1, 9999)} {st}, {city}, {c}",
                "country": c
            })
            
            s3_id = f"S3_{split.upper()}_{s3_counter:06d}"
            s3_counter += 1
            b_name3 = random.choice(CORE_NAMES) + f" Unrelated {random.randint(100, 999)}"
            c3 = random.choice(COUNTRIES)
            city3 = random.choice(CITIES[c3])
            st3 = random.choice(STREETS)
            s3_records.append({
                "entity_id": s3_id,
                "business_name": f"{b_name3} {random.choice(LEGAL_SUFFIX_MAP[c3])}",
                "business_address": f"{random.randint(1, 9999)} {st3}, {city3}, {c3}",
                "country": c3
            })

        random.shuffle(s2_records)
        random.shuffle(s3_records)

        df_s1 = pd.DataFrame(s1_records)
        df_s2 = pd.DataFrame(s2_records)
        df_s3 = pd.DataFrame(s3_records)
        
        target_dir = f"data/{split}"
        df_s1.to_csv(f"{target_dir}/{split}_source1.tsv", sep="\t", index=False)
        df_s2.to_csv(f"{target_dir}/{split}_source2.tsv", sep="\t", index=False)
        df_s3.to_csv(f"{target_dir}/{split}_source3.tsv", sep="\t", index=False)
        
        if split == "train":
            df_gt = pd.DataFrame(ground_truth)
            df_gt.to_csv(f"{target_dir}/{split}_ground_truth.tsv", sep="\t", index=False)
            print(f"Generated Train: S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)}, GT={len(df_gt)}")
        else:
            print(f"Generated Test: S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)}")

if __name__ == "__main__":
    generate_dataset()
