"""
dataset.py — Semantic convergence dataset (24 factors, 12 sentences each = 288 total)

Surface-family split policy (Instructions §4, §7):
  train : F1 + F2  (canonical + lexical paraphrase)
  val   : F3       (syntactic paraphrase)
  test  : F4 + F5  (definition-like + richer context)
"""

import json
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Factor registry: 24 factors × 6 domains (§5)
# Each factor has 2 F1, 2 F2, 4 F3, 2 F4, 2 F5 = 12 sentences
# ---------------------------------------------------------------------------

FACTORS = [

    # ── DOMAIN A ── Scalar temperature ─────────────────────────────────────

    {
        "factor_id": "A1", "domain_id": "A",
        "canonical_meaning": "the object is hot",
        "hard_negative_factor_ids": ["A3", "A2", "A4"],
        "sentences": [
            {"text": "The object is hot.",                                          "family": "F1", "split": "train"},
            {"text": "The thing is hot.",                                           "family": "F1", "split": "train"},
            {"text": "The item has a high temperature.",                            "family": "F2", "split": "train"},
            {"text": "The object exhibits intense heat.",                           "family": "F2", "split": "train"},
            {"text": "What characterizes the object is its high temperature.",      "family": "F3", "split": "val"},
            {"text": "Hot is how the object would be described.",                   "family": "F3", "split": "val"},
            {"text": "The object, notably hot, is defined by its heat.",            "family": "F3", "split": "val"},
            {"text": "It is the heat of the object that makes it stand out.",       "family": "F3", "split": "val"},
            {"text": "Touching it would cause immediate discomfort from heat.",     "family": "F4", "split": "test"},
            {"text": "It radiates enough heat to be very uncomfortable.",           "family": "F4", "split": "test"},
            {"text": "Upon contact, one immediately notices the intense heat.",     "family": "F5", "split": "test"},
            {"text": "When touched, the surface feels noticeably hot.",             "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "A2", "domain_id": "A",
        "canonical_meaning": "the object is cold",
        "hard_negative_factor_ids": ["A4", "A3", "A1"],
        "sentences": [
            {"text": "The object is cold.",                                         "family": "F1", "split": "train"},
            {"text": "The thing is cold.",                                          "family": "F1", "split": "train"},
            {"text": "The item has a low temperature.",                             "family": "F2", "split": "train"},
            {"text": "The object registers a cold temperature.",                    "family": "F2", "split": "train"},
            {"text": "Cold is how the object would be described.",                  "family": "F3", "split": "val"},
            {"text": "What characterizes the object is its low temperature.",       "family": "F3", "split": "val"},
            {"text": "The object, notably cold, is defined by its chill.",          "family": "F3", "split": "val"},
            {"text": "It is the coldness of the object that is noticeable.",        "family": "F3", "split": "val"},
            {"text": "Touching it produces a strong cold sensation.",               "family": "F4", "split": "test"},
            {"text": "It draws heat from the hand upon contact.",                   "family": "F4", "split": "test"},
            {"text": "Upon contact, the cold is immediately apparent.",             "family": "F5", "split": "test"},
            {"text": "When touched, the surface sends a chill through one's fingers.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "A3", "domain_id": "A",
        "canonical_meaning": "the object is warm",
        "hard_negative_factor_ids": ["A4", "A1", "A2"],
        "sentences": [
            {"text": "The object is warm.",                                         "family": "F1", "split": "train"},
            {"text": "The thing is warm.",                                          "family": "F1", "split": "train"},
            {"text": "The item has a moderate warmth.",                             "family": "F2", "split": "train"},
            {"text": "The object has a pleasantly elevated temperature.",           "family": "F2", "split": "train"},
            {"text": "Warm is how the object would be described.",                  "family": "F3", "split": "val"},
            {"text": "What characterizes the object is its gentle warmth.",         "family": "F3", "split": "val"},
            {"text": "The object, which is warm, gives off gentle heat.",           "family": "F3", "split": "val"},
            {"text": "It is the warmth of the object that makes it comfortable.",   "family": "F3", "split": "val"},
            {"text": "Touching it gives a pleasant warm sensation.",                "family": "F4", "split": "test"},
            {"text": "It gives off mild heat without being uncomfortable.",         "family": "F4", "split": "test"},
            {"text": "Upon contact, one feels a gentle, comfortable warmth.",       "family": "F5", "split": "test"},
            {"text": "When touched, it feels pleasantly warm but not hot.",         "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "A4", "domain_id": "A",
        "canonical_meaning": "the object is cool",
        "hard_negative_factor_ids": ["A3", "A2", "A1"],
        "sentences": [
            {"text": "The object is cool.",                                         "family": "F1", "split": "train"},
            {"text": "The thing is cool.",                                          "family": "F1", "split": "train"},
            {"text": "The item is slightly below room temperature.",                "family": "F2", "split": "train"},
            {"text": "The object has a mildly low temperature.",                    "family": "F2", "split": "train"},
            {"text": "Cool is how the object would be described.",                  "family": "F3", "split": "val"},
            {"text": "What characterizes the object is its mild coolness.",         "family": "F3", "split": "val"},
            {"text": "The object, which is cool, is mildly chilled.",               "family": "F3", "split": "val"},
            {"text": "It is the slight coolness that characterizes the object.",    "family": "F3", "split": "val"},
            {"text": "Touching it produces a mild cool sensation.",                 "family": "F4", "split": "test"},
            {"text": "It feels cooler than the surrounding air.",                   "family": "F4", "split": "test"},
            {"text": "Upon contact, one notices a mild, refreshing coolness.",      "family": "F5", "split": "test"},
            {"text": "When touched, it feels slightly cool without being cold.",    "family": "F5", "split": "test"},
        ],
    },

    # ── DOMAIN B ── Scalar size ─────────────────────────────────────────────

    {
        "factor_id": "B1", "domain_id": "B",
        "canonical_meaning": "the object is large",
        "hard_negative_factor_ids": ["B2", "B4"],
        "sentences": [
            {"text": "The object is large.",                                        "family": "F1", "split": "train"},
            {"text": "The thing is large.",                                         "family": "F1", "split": "train"},
            {"text": "The item is big in size.",                                    "family": "F2", "split": "train"},
            {"text": "The object takes up considerable space.",                     "family": "F2", "split": "train"},
            {"text": "Large is how the object would be described.",                 "family": "F3", "split": "val"},
            {"text": "What the object is known for is being large.",                "family": "F3", "split": "val"},
            {"text": "The object, which is large, stands out for its size.",        "family": "F3", "split": "val"},
            {"text": "It is the large size of the object that is notable.",         "family": "F3", "split": "val"},
            {"text": "It is noticeably bigger than average.",                       "family": "F4", "split": "test"},
            {"text": "You would call it large compared to similar objects.",        "family": "F4", "split": "test"},
            {"text": "When placed next to a typical object, it appears quite large.", "family": "F5", "split": "test"},
            {"text": "One immediately notices the large size when looking at the object.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "B2", "domain_id": "B",
        "canonical_meaning": "the object is small",
        "hard_negative_factor_ids": ["B1", "B4"],
        "sentences": [
            {"text": "The object is small.",                                        "family": "F1", "split": "train"},
            {"text": "The thing is small.",                                         "family": "F1", "split": "train"},
            {"text": "The item is little in size.",                                 "family": "F2", "split": "train"},
            {"text": "The object takes up very little space.",                      "family": "F2", "split": "train"},
            {"text": "Small is how the object would be described.",                 "family": "F3", "split": "val"},
            {"text": "What the object is known for is being small.",                "family": "F3", "split": "val"},
            {"text": "The object, which is small, is compact in size.",             "family": "F3", "split": "val"},
            {"text": "It is the small size of the object that is notable.",         "family": "F3", "split": "val"},
            {"text": "It is noticeably smaller than average.",                      "family": "F4", "split": "test"},
            {"text": "You would call it small compared to similar objects.",        "family": "F4", "split": "test"},
            {"text": "When placed next to a typical object, it appears quite small.", "family": "F5", "split": "test"},
            {"text": "One immediately notices the small size when looking at the object.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "B3", "domain_id": "B",
        "canonical_meaning": "the object is huge",
        "hard_negative_factor_ids": ["B1", "B2"],
        "sentences": [
            {"text": "The object is huge.",                                         "family": "F1", "split": "train"},
            {"text": "The thing is huge.",                                          "family": "F1", "split": "train"},
            {"text": "The item is enormous in size.",                               "family": "F2", "split": "train"},
            {"text": "The object is extremely large.",                              "family": "F2", "split": "train"},
            {"text": "Huge is how the object would be described.",                  "family": "F3", "split": "val"},
            {"text": "What the object is known for is being huge.",                 "family": "F3", "split": "val"},
            {"text": "The object, which is huge, is far larger than typical.",      "family": "F3", "split": "val"},
            {"text": "It is the immense size of the object that stands out.",       "family": "F3", "split": "val"},
            {"text": "It is vastly bigger than comparable objects.",                "family": "F4", "split": "test"},
            {"text": "Few things are as large as this object.",                     "family": "F4", "split": "test"},
            {"text": "When placed next to a typical object, it is dramatically larger.", "family": "F5", "split": "test"},
            {"text": "The sheer hugeness of the object is immediately apparent.",   "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "B4", "domain_id": "B",
        "canonical_meaning": "the object is tiny",
        "hard_negative_factor_ids": ["B2", "B1"],
        "sentences": [
            {"text": "The object is tiny.",                                         "family": "F1", "split": "train"},
            {"text": "The thing is tiny.",                                          "family": "F1", "split": "train"},
            {"text": "The item is minuscule in size.",                              "family": "F2", "split": "train"},
            {"text": "The object is extremely small.",                              "family": "F2", "split": "train"},
            {"text": "Tiny is how the object would be described.",                  "family": "F3", "split": "val"},
            {"text": "What the object is known for is being tiny.",                 "family": "F3", "split": "val"},
            {"text": "The object, which is tiny, is far smaller than typical.",     "family": "F3", "split": "val"},
            {"text": "It is the very small size of the object that stands out.",    "family": "F3", "split": "val"},
            {"text": "It is drastically smaller than comparable objects.",          "family": "F4", "split": "test"},
            {"text": "Few things are as small as this object.",                     "family": "F4", "split": "test"},
            {"text": "When placed next to a typical object, it appears very tiny.", "family": "F5", "split": "test"},
            {"text": "The tiny size of the object is immediately apparent.",        "family": "F5", "split": "test"},
        ],
    },

    # ── DOMAIN C ── Taxonomic / refinement ─────────────────────────────────

    {
        "factor_id": "C1", "domain_id": "C",
        "canonical_meaning": "the animal is a dog",
        "hard_negative_factor_ids": ["C3", "C2"],
        "sentences": [
            {"text": "The animal is a dog.",                                        "family": "F1", "split": "train"},
            {"text": "The creature is a dog.",                                      "family": "F1", "split": "train"},
            {"text": "It is a domesticated canine.",                                "family": "F2", "split": "train"},
            {"text": "The animal belongs to the dog species.",                      "family": "F2", "split": "train"},
            {"text": "What the animal is, specifically, is a dog.",                 "family": "F3", "split": "val"},
            {"text": "A dog is what this animal is.",                               "family": "F3", "split": "val"},
            {"text": "The animal, which is a dog, is domesticated.",                "family": "F3", "split": "val"},
            {"text": "It is the fact that the animal is a dog that matters.",       "family": "F3", "split": "val"},
            {"text": "This is the kind of animal that barks and is kept as a pet.", "family": "F4", "split": "test"},
            {"text": "It is the most common household canine companion.",           "family": "F4", "split": "test"},
            {"text": "As a dog, this animal is one of the most familiar domestic pets.", "family": "F5", "split": "test"},
            {"text": "This animal belongs to the species commonly known as the domestic dog.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "C2", "domain_id": "C",
        "canonical_meaning": "the animal is a canine",
        "hard_negative_factor_ids": ["C3", "C1"],
        "sentences": [
            {"text": "The animal is a canine.",                                     "family": "F1", "split": "train"},
            {"text": "The creature is a canine.",                                   "family": "F1", "split": "train"},
            {"text": "It belongs to the dog family.",                               "family": "F2", "split": "train"},
            {"text": "The animal is a member of the canid family.",                 "family": "F2", "split": "train"},
            {"text": "What the animal is, specifically, is a canine.",              "family": "F3", "split": "val"},
            {"text": "A canine is what this animal is.",                            "family": "F3", "split": "val"},
            {"text": "The animal, which is a canine, belongs to the dog family.",   "family": "F3", "split": "val"},
            {"text": "It is the canine nature of the animal that defines it.",      "family": "F3", "split": "val"},
            {"text": "This is an animal from the dog family, broader than just dogs.", "family": "F4", "split": "test"},
            {"text": "It is a member of the taxonomic group that includes wolves and foxes.", "family": "F4", "split": "test"},
            {"text": "As a canine, this animal belongs to the same family as wolves.", "family": "F5", "split": "test"},
            {"text": "This animal is classified within the canid family, making it a canine.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "C3", "domain_id": "C",
        "canonical_meaning": "the animal is a mammal",
        "hard_negative_factor_ids": ["C1"],
        "sentences": [
            {"text": "The animal is a mammal.",                                     "family": "F1", "split": "train"},
            {"text": "The creature is a mammal.",                                   "family": "F1", "split": "train"},
            {"text": "It belongs to the mammal class.",                             "family": "F2", "split": "train"},
            {"text": "The animal is a mammalian creature.",                         "family": "F2", "split": "train"},
            {"text": "What the animal is, specifically, is a mammal.",              "family": "F3", "split": "val"},
            {"text": "A mammal is what this animal is.",                            "family": "F3", "split": "val"},
            {"text": "The animal, which is a mammal, nurses its young.",            "family": "F3", "split": "val"},
            {"text": "It is the mammalian classification that defines this animal.", "family": "F3", "split": "val"},
            {"text": "This animal is warm-blooded and nurses its young with milk.", "family": "F4", "split": "test"},
            {"text": "It is the type of animal that has hair and feeds young with milk.", "family": "F4", "split": "test"},
            {"text": "As a mammal, this animal is warm-blooded and raises young on milk.", "family": "F5", "split": "test"},
            {"text": "This creature belongs to the class of animals that includes humans and whales.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "C4", "domain_id": "C",
        "canonical_meaning": "the dog is a working dog",
        "hard_negative_factor_ids": ["C1", "C2"],
        "sentences": [
            {"text": "The dog is a working dog.",                                   "family": "F1", "split": "train"},
            {"text": "This dog is a working dog.",                                  "family": "F1", "split": "train"},
            {"text": "It is trained to perform tasks for people.",                  "family": "F2", "split": "train"},
            {"text": "The animal is a task-trained dog.",                           "family": "F2", "split": "train"},
            {"text": "What defines this dog is its working role.",                  "family": "F3", "split": "val"},
            {"text": "A working dog is what this dog is.",                          "family": "F3", "split": "val"},
            {"text": "The dog, which is a working dog, performs practical tasks.",  "family": "F3", "split": "val"},
            {"text": "It is its working role that sets this dog apart.",            "family": "F3", "split": "val"},
            {"text": "This dog is trained for specific tasks rather than just companionship.", "family": "F4", "split": "test"},
            {"text": "It performs functions like herding, guarding, or assisting people.", "family": "F4", "split": "test"},
            {"text": "As a working dog, it is trained and used for specific practical tasks.", "family": "F5", "split": "test"},
            {"text": "Unlike pet dogs, this dog serves in a professional or service capacity.", "family": "F5", "split": "test"},
        ],
    },

    # ── DOMAIN D ── Structured object / function ────────────────────────────

    {
        "factor_id": "D1", "domain_id": "D",
        "canonical_meaning": "the vehicle is an electric passenger road vehicle",
        "hard_negative_factor_ids": ["D2"],
        "sentences": [
            {"text": "The vehicle is an electric passenger road vehicle.",          "family": "F1", "split": "train"},
            {"text": "This vehicle is an electric car for passengers.",             "family": "F1", "split": "train"},
            {"text": "It is a road-going passenger vehicle powered by electricity.", "family": "F2", "split": "train"},
            {"text": "The car carries people and runs on battery power.",           "family": "F2", "split": "train"},
            {"text": "What defines this vehicle is that it is electric and carries passengers.", "family": "F3", "split": "val"},
            {"text": "An electric passenger vehicle is what this is.",              "family": "F3", "split": "val"},
            {"text": "The vehicle, which is electric and passenger-carrying, runs on batteries.", "family": "F3", "split": "val"},
            {"text": "It is the electric propulsion and passenger function that define this vehicle.", "family": "F3", "split": "val"},
            {"text": "This vehicle runs on electricity and is designed to carry people.", "family": "F4", "split": "test"},
            {"text": "It is a battery-powered car meant to transport passengers.",  "family": "F4", "split": "test"},
            {"text": "As an electric passenger vehicle, it carries people using battery-powered motors.", "family": "F5", "split": "test"},
            {"text": "This vehicle transports passengers along roads using stored electrical energy.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "D2", "domain_id": "D",
        "canonical_meaning": "the vehicle is used to transport goods",
        "hard_negative_factor_ids": ["D1", "D3"],
        "sentences": [
            {"text": "The vehicle is a cargo vehicle.",                             "family": "F1", "split": "train"},
            {"text": "This vehicle carries cargo.",                                 "family": "F1", "split": "train"},
            {"text": "It is meant to carry goods rather than passengers.",          "family": "F2", "split": "train"},
            {"text": "The vehicle is designed for transporting cargo.",             "family": "F2", "split": "train"},
            {"text": "What defines this vehicle is its cargo-carrying purpose.",    "family": "F3", "split": "val"},
            {"text": "A cargo vehicle is what this is.",                            "family": "F3", "split": "val"},
            {"text": "The vehicle, which is designed for cargo, moves goods rather than people.", "family": "F3", "split": "val"},
            {"text": "It is the function of carrying goods that defines this vehicle.", "family": "F3", "split": "val"},
            {"text": "This vehicle is built for moving freight rather than people.", "family": "F4", "split": "test"},
            {"text": "It is a type of vehicle intended to transport goods from place to place.", "family": "F4", "split": "test"},
            {"text": "As a cargo vehicle, it is designed to move goods rather than passengers.", "family": "F5", "split": "test"},
            {"text": "This vehicle's primary function is the transportation of freight and goods.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "D3", "domain_id": "D",
        "canonical_meaning": "the vehicle is a compact car",
        "hard_negative_factor_ids": ["D2"],
        "sentences": [
            {"text": "The vehicle is a compact car.",                               "family": "F1", "split": "train"},
            {"text": "This automobile is a compact car.",                           "family": "F1", "split": "train"},
            {"text": "It is a small passenger car.",                                "family": "F2", "split": "train"},
            {"text": "The car is built in a compact size class.",                   "family": "F2", "split": "train"},
            {"text": "What defines this vehicle is being a compact car.",           "family": "F3", "split": "val"},
            {"text": "A compact car is what this vehicle is.",                      "family": "F3", "split": "val"},
            {"text": "The car, which is compact in size, carries passengers efficiently.", "family": "F3", "split": "val"},
            {"text": "It is the compact size class that defines this car.",         "family": "F3", "split": "val"},
            {"text": "This car is notably smaller than a standard full-size vehicle.", "family": "F4", "split": "test"},
            {"text": "It is a small passenger automobile, efficient for urban use.", "family": "F4", "split": "test"},
            {"text": "As a compact car, it occupies less space than standard sedans.", "family": "F5", "split": "test"},
            {"text": "This automobile is classified as compact due to its relatively small size.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "D4", "domain_id": "D",
        "canonical_meaning": "the software system is a distributed database",
        "hard_negative_factor_ids": [],
        "sentences": [
            {"text": "The software system is a distributed database.",              "family": "F1", "split": "train"},
            {"text": "This system is a distributed database.",                      "family": "F1", "split": "train"},
            {"text": "It stores data across multiple machines.",                    "family": "F2", "split": "train"},
            {"text": "The database system is distributed over several nodes.",      "family": "F2", "split": "train"},
            {"text": "What defines this system is being a distributed database.",   "family": "F3", "split": "val"},
            {"text": "A distributed database is what this software is.",            "family": "F3", "split": "val"},
            {"text": "The system, which is a distributed database, runs on multiple nodes.", "family": "F3", "split": "val"},
            {"text": "It is the distributed nature of this database that sets it apart.", "family": "F3", "split": "val"},
            {"text": "This software stores and manages data across multiple servers.", "family": "F4", "split": "test"},
            {"text": "It is a database that does not rely on a single machine for storage.", "family": "F4", "split": "test"},
            {"text": "As a distributed database, it splits data across many nodes for reliability.", "family": "F5", "split": "test"},
            {"text": "This database manages data across a cluster of machines rather than a single server.", "family": "F5", "split": "test"},
        ],
    },

    # ── DOMAIN E ── Human-designed object refinements ───────────────────────

    {
        "factor_id": "E1", "domain_id": "E",
        "canonical_meaning": "the footwear is a waterproof hiking boot",
        "hard_negative_factor_ids": [],
        "sentences": [
            {"text": "The footwear is a waterproof hiking boot.",                   "family": "F1", "split": "train"},
            {"text": "This shoe is a waterproof hiking boot.",                      "family": "F1", "split": "train"},
            {"text": "It is a hiking boot that keeps water out.",                   "family": "F2", "split": "train"},
            {"text": "The trail boot resists water entry.",                         "family": "F2", "split": "train"},
            {"text": "What defines this footwear is being a waterproof hiking boot.", "family": "F3", "split": "val"},
            {"text": "A waterproof hiking boot is what this is.",                   "family": "F3", "split": "val"},
            {"text": "The boot, which is designed for hiking, also repels water.",  "family": "F3", "split": "val"},
            {"text": "It is the combination of waterproofing and hiking design that defines this boot.", "family": "F3", "split": "val"},
            {"text": "This boot is made for walking trails while keeping feet dry.", "family": "F4", "split": "test"},
            {"text": "It is outdoor footwear designed for wet hiking conditions.",  "family": "F4", "split": "test"},
            {"text": "As a waterproof hiking boot, it is built for trail use in wet environments.", "family": "F5", "split": "test"},
            {"text": "This boot protects feet from moisture while providing support for hiking.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "E2", "domain_id": "E",
        "canonical_meaning": "the chair is an ergonomic office chair",
        "hard_negative_factor_ids": [],
        "sentences": [
            {"text": "The chair is an ergonomic office chair.",                     "family": "F1", "split": "train"},
            {"text": "This seat is an ergonomic office chair.",                     "family": "F1", "split": "train"},
            {"text": "It is an office chair designed for ergonomic support.",       "family": "F2", "split": "train"},
            {"text": "The chair is made for desk work and body comfort.",           "family": "F2", "split": "train"},
            {"text": "What defines this chair is being ergonomic and for office use.", "family": "F3", "split": "val"},
            {"text": "An ergonomic office chair is what this is.",                  "family": "F3", "split": "val"},
            {"text": "The chair, which is ergonomic, is made for office settings.", "family": "F3", "split": "val"},
            {"text": "It is the ergonomic design for office use that defines this chair.", "family": "F3", "split": "val"},
            {"text": "This chair is designed to support posture during long hours at a desk.", "family": "F4", "split": "test"},
            {"text": "It is a work seat engineered to reduce physical strain.",     "family": "F4", "split": "test"},
            {"text": "As an ergonomic office chair, it supports the body during desk work.", "family": "F5", "split": "test"},
            {"text": "This chair provides ergonomic support suited for office environments.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "E3", "domain_id": "E",
        "canonical_meaning": "the building is a residential apartment building",
        "hard_negative_factor_ids": ["E4"],
        "sentences": [
            {"text": "The building is a residential apartment building.",           "family": "F1", "split": "train"},
            {"text": "This structure is a residential apartment building.",         "family": "F1", "split": "train"},
            {"text": "It is a structure containing apartments for people to live in.", "family": "F2", "split": "train"},
            {"text": "The building is used for housing multiple households.",       "family": "F2", "split": "train"},
            {"text": "What defines this building is being a residential apartment building.", "family": "F3", "split": "val"},
            {"text": "A residential apartment building is what this structure is.", "family": "F3", "split": "val"},
            {"text": "The building, which contains apartments, is used for residential living.", "family": "F3", "split": "val"},
            {"text": "It is the residential apartment function that defines this building.", "family": "F3", "split": "val"},
            {"text": "This building contains multiple dwelling units for people to live in.", "family": "F4", "split": "test"},
            {"text": "It is a multi-unit structure where people reside in separate apartments.", "family": "F4", "split": "test"},
            {"text": "As a residential apartment building, it provides housing to multiple households.", "family": "F5", "split": "test"},
            {"text": "This structure is built to serve as home to many people in separate apartment units.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "E4", "domain_id": "E",
        "canonical_meaning": "the building is a high-rise apartment building",
        "hard_negative_factor_ids": ["E3"],
        "sentences": [
            {"text": "The building is a high-rise apartment building.",             "family": "F1", "split": "train"},
            {"text": "This structure is a high-rise apartment building.",           "family": "F1", "split": "train"},
            {"text": "It is a tall residential structure containing apartments.",   "family": "F2", "split": "train"},
            {"text": "This apartment building rises many stories high.",            "family": "F2", "split": "train"},
            {"text": "What defines this building is being a tall apartment building.", "family": "F3", "split": "val"},
            {"text": "A high-rise apartment building is what this structure is.",   "family": "F3", "split": "val"},
            {"text": "The building, which is a high-rise, contains residential apartments.", "family": "F3", "split": "val"},
            {"text": "It is the height and apartment function that define this building.", "family": "F3", "split": "val"},
            {"text": "This tall building contains many floors of apartment residences.", "family": "F4", "split": "test"},
            {"text": "It is a very tall structure where many families live in stacked apartments.", "family": "F4", "split": "test"},
            {"text": "As a high-rise apartment building, it houses many residents across many floors.", "family": "F5", "split": "test"},
            {"text": "This multi-story structure contains apartments across its many levels.", "family": "F5", "split": "test"},
        ],
    },

    # ── DOMAIN F ── Institutional / legal ──────────────────────────────────

    {
        "factor_id": "F1", "domain_id": "F",
        "canonical_meaning": "the institution is a public research university",
        "hard_negative_factor_ids": ["F2"],
        "sentences": [
            {"text": "The institution is a public research university.",            "family": "F1", "split": "train"},
            {"text": "This organization is a public research university.",          "family": "F1", "split": "train"},
            {"text": "It is a publicly funded university with a strong research mission.", "family": "F2", "split": "train"},
            {"text": "The university belongs to the public sector and focuses on research.", "family": "F2", "split": "train"},
            {"text": "What defines this institution is being a public research university.", "family": "F3", "split": "val"},
            {"text": "A public research university is what this institution is.",   "family": "F3", "split": "val"},
            {"text": "The university, which is public and research-focused, receives state funding.", "family": "F3", "split": "val"},
            {"text": "It is the public funding and research mission that define this university.", "family": "F3", "split": "val"},
            {"text": "This is a state-funded university where significant research takes place.", "family": "F4", "split": "test"},
            {"text": "It is a publicly supported higher education institution with major research programs.", "family": "F4", "split": "test"},
            {"text": "As a public research university, it is funded by the state and produces research.", "family": "F5", "split": "test"},
            {"text": "This institution is publicly funded and known for its strong research activities.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "F2", "domain_id": "F",
        "canonical_meaning": "the institution is a public university",
        "hard_negative_factor_ids": ["F1"],
        "sentences": [
            {"text": "The institution is a public university.",                     "family": "F1", "split": "train"},
            {"text": "This organization is a public university.",                   "family": "F1", "split": "train"},
            {"text": "It is a university funded by the state or public sector.",    "family": "F2", "split": "train"},
            {"text": "The university belongs to the public education system.",      "family": "F2", "split": "train"},
            {"text": "What defines this institution is being a public university.", "family": "F3", "split": "val"},
            {"text": "A public university is what this institution is.",            "family": "F3", "split": "val"},
            {"text": "The university, which is publicly funded, grants university degrees.", "family": "F3", "split": "val"},
            {"text": "It is the public funding that defines this university.",      "family": "F3", "split": "val"},
            {"text": "This university is funded by government rather than private sources.", "family": "F4", "split": "test"},
            {"text": "It is a state-supported higher education institution.",       "family": "F4", "split": "test"},
            {"text": "As a public university, it is funded by public money and open to students broadly.", "family": "F5", "split": "test"},
            {"text": "This institution receives state funding and operates as a public higher education provider.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "F3", "domain_id": "F",
        "canonical_meaning": "the document is a fixed-term employment contract",
        "hard_negative_factor_ids": ["F4"],
        "sentences": [
            {"text": "The document is a fixed-term employment contract.",           "family": "F1", "split": "train"},
            {"text": "This agreement is a fixed-term employment contract.",         "family": "F1", "split": "train"},
            {"text": "It is an employment agreement with a defined end date.",      "family": "F2", "split": "train"},
            {"text": "This contract hires a worker for a limited period.",          "family": "F2", "split": "train"},
            {"text": "What defines this document is being a fixed-term employment contract.", "family": "F3", "split": "val"},
            {"text": "A fixed-term employment contract is what this document is.",  "family": "F3", "split": "val"},
            {"text": "The contract, which covers employment, specifies a fixed end date.", "family": "F3", "split": "val"},
            {"text": "It is the fixed duration of this employment contract that defines it.", "family": "F3", "split": "val"},
            {"text": "This document formalizes a job arrangement that will end on a specific date.", "family": "F4", "split": "test"},
            {"text": "It is a contract that employs someone for a set period, after which it expires.", "family": "F4", "split": "test"},
            {"text": "As a fixed-term contract, it establishes employment that ends on a set date.", "family": "F5", "split": "test"},
            {"text": "This employment contract differs from permanent ones in that it has a defined expiration.", "family": "F5", "split": "test"},
        ],
    },

    {
        "factor_id": "F4", "domain_id": "F",
        "canonical_meaning": "the document is an employment contract",
        "hard_negative_factor_ids": ["F3"],
        "sentences": [
            {"text": "The document is an employment contract.",                     "family": "F1", "split": "train"},
            {"text": "This agreement is an employment contract.",                   "family": "F1", "split": "train"},
            {"text": "It is an agreement governing a work relationship.",           "family": "F2", "split": "train"},
            {"text": "The document formalizes a job arrangement.",                  "family": "F2", "split": "train"},
            {"text": "What defines this document is being an employment contract.", "family": "F3", "split": "val"},
            {"text": "An employment contract is what this document is.",            "family": "F3", "split": "val"},
            {"text": "The contract, which governs employment, sets out the terms of work.", "family": "F3", "split": "val"},
            {"text": "It is the employment relationship that this contract governs.", "family": "F3", "split": "val"},
            {"text": "This document sets out the formal terms under which someone is employed.", "family": "F4", "split": "test"},
            {"text": "It is a legal agreement establishing the conditions of a paid work relationship.", "family": "F4", "split": "test"},
            {"text": "As an employment contract, it formalizes the agreement between employer and employee.", "family": "F5", "split": "test"},
            {"text": "This document defines the rights and obligations of both employer and worker.", "family": "F5", "split": "test"},
        ],
    },
]


def build_dataframe() -> pd.DataFrame:
    """Flatten FACTORS into a sentence-level DataFrame."""
    rows = []
    for factor in FACTORS:
        for i, sent in enumerate(factor["sentences"]):
            rows.append({
                "sentence_id": f"{factor['factor_id']}_{i:02d}",
                "factor_id": factor["factor_id"],
                "domain_id": factor["domain_id"],
                "canonical_meaning": factor["canonical_meaning"],
                "surface_family": sent["family"],
                "split": sent["split"],
                "sentence_text": sent["text"],
                "hard_negative_factor_ids": json.dumps(factor["hard_negative_factor_ids"]),
            })
    df = pd.DataFrame(rows)
    df["language"] = "en"
    return df


def build_full_dataframe() -> pd.DataFrame:
    """Return combined English + multilingual DataFrame (480 rows total).

    English sentences have language='en'; multilingual sentences have
    language in ['de','es','ar','zh'] and split='ml'.
    """
    import ml_sentences as ml
    df_en = build_dataframe()
    df_ml = ml.build_ml_dataframe()
    # Align columns: df_en has extra columns not in df_ml; fill with NaN
    df_combined = pd.concat([df_en, df_ml], ignore_index=True, sort=False)
    df_combined.reset_index(drop=True, inplace=True)
    return df_combined


def get_factor_index() -> dict:
    """Map factor_id → integer index (0-based)."""
    return {f["factor_id"]: i for i, f in enumerate(FACTORS)}


def get_domain_index() -> dict:
    """Map factor_id → domain integer index."""
    domains = sorted(set(f["domain_id"] for f in FACTORS))
    domain_to_int = {d: i for i, d in enumerate(domains)}
    return {f["factor_id"]: domain_to_int[f["domain_id"]] for f in FACTORS}


if __name__ == "__main__":
    df = build_dataframe()
    out = Path("dataset.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} sentences to {out}")
    print(df.groupby(["split", "surface_family"]).size().to_string())
    print(f"\nFactors: {df['factor_id'].nunique()}  |  Domains: {df['domain_id'].nunique()}")
