from .engine import Sample_prob
def run_model(monitors=[], observes=[], observed_values=[], ninter=1000):
    sp = Sample_prob()
    samp = sp.sample(monitors, observes, observed_values, ninter)
    return samp