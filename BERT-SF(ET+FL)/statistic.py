import os
from collections import Counter


def read_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        res = [x.strip() for x in f.readlines()]
    return res


# 1. get_data_distribution
train_raw_path = 'data/snips/train'
train_raw_label = read_txt(os.path.join(train_raw_path, 'intent_seq.out'))
intent_labels = [x.split()[0] for x in train_raw_label]
intent_dict = dict(Counter(intent_labels))
intent_distrib = {k: round(v / (sum(intent_dict.values())), 4) for k, v in intent_dict.items()}
print(intent_dict)
print(intent_distrib)

# 2.0 key word extraction
keyword_dict = {"atis_flight": ["flight", "trip"], "atis_airfare": ["fare", "cost", "price", "ticket", "how much"],
                "air_line": ["airline"],
                "atis_ground_service": ["ground", "service"], "atis_quantity": ["how many"],
                "air_city": ["city", "cities"], "atis_flight#atis_airfare": ["fare"],
                "atis_abbreviation": ["what is", "what does ... mean"], "atis_aircraft": ["aircraft", "plane"],
                "aits_distance": ["how long", "how far", "distance"],
                "atis_ground_fare": ["how much", "price", "cost"], "atis_capacity": ["how many", "capacity"],
                "atis_flight_time": ["time", "schedule"],
                "atis_meal": ["meal"], "atis_aircraft#atis_flight#atis_flight_no": ["aircraft", "flight", "number"],
                "atis_flight_no": ["number"],
                "atis_restriction": ["restriction"], "atis_airport": ["airport"],
                "atis_airline#atis_flight_no": ["airline", "number"], "atis_cheapest": ["cheapest", "fare"],
                "atis_ground_service#atis_ground_fare": ["ground", "service", "fare"]
                }
keyword_list = list(set([x for y in keyword_dict.values() for x in y]))

# 2. get_specific_datac
train_raw_input = read_txt(os.path.join(train_raw_path, 'intent_seq.in'))
for s, label in zip(train_raw_input, train_raw_label):
    if label.split()[0] == "atis_cheapest":
        print(s)
