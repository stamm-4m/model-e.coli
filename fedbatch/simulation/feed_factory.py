from fedbatch.simulation.feed_profile import (
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