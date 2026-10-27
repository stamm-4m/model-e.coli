"""
Nanobody-based Antivenom Production with E. coli Reactor Simulation 

Creates a feed profile based on the configuration provided in the `cfg` dictionary. 
The feed profile can be of various types, including constant, linear, exponential, and on-off feed profiles. 
See feed_profile module for the implementation of each feed type.

Author: Juan Camilo Castaño Sanchez
Email: jcastano-san@insa-toulose.fr
Date: 01/09/2026
"""

from src.knowledge_based_workflow.model.auxiliar.feed_profile import (
    ConstantFeed,
    LinearFeed,
    ExponentialFeed,
    OnOffFeed,
    OnOffFeed_Linear
)

def create_feed(cfg):
    feed_type = cfg["type"]

    if feed_type == "constant":
        return ConstantFeed(F0=cfg["F0"])

    if feed_type == "linear":
        return LinearFeed(F0=cfg["F0"], slope=cfg["slope"])

    if feed_type == "exponential":
        return ExponentialFeed(F0=cfg["F0"], k=cfg["k"])
    
    if feed_type == "OnOffFeed":
        return OnOffFeed(intervals=cfg["value"])
    
    if feed_type == "OnOffFeed_Linear":
        return OnOffFeed_Linear(intervals=cfg["value"])

    raise ValueError(f"Unknown feed type: {feed_type}")