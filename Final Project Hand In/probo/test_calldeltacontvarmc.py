from probo.marketdata import MarketData
from probo.payoff import VanillaPayoff, call_payoff
from probo.engine import MonteCarloEngine, DeltaControlVariatePricerCall
from probo.facade import OptionFacade

## Set up the market data
spot = 41.0
rate = 0.08
volatility = 0.30
dividend = 0.0
thedata = MarketData(rate, spot, volatility, dividend)

## Set up the option
expiry = 0.25
strike = 40.0
thecall = VanillaPayoff(expiry, strike, call_payoff)

## Set up Delta Control Variate Monte Carlo
nreps = 100000
steps = 1
pricer = DeltaControlVariatePricerCall
mcengine = MonteCarloEngine(nreps, steps, pricer)

## Calculate the price
option1 = OptionFacade(thecall, mcengine, thedata)
price1, stderr1 = option1.price()
print("The call price via Delta Control Variate Monte Carlo is: {0:.4f} and the standard error is {1:.4f}.".format(price1, stderr1))



