from probo.marketdata import MarketData
from probo.payoff import VanillaPayoff, put_payoff
from probo.engine import MonteCarloEngine, DeltaControlVariatePricerPut
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
theput = VanillaPayoff(expiry, strike, put_payoff)

## Set up Delta Control Variate Monte Carlo
nreps = 100000
steps = 1
pricer = DeltaControlVariatePricerPut
mcengine = MonteCarloEngine(nreps, steps, pricer)

## Calculate the price
option2 = OptionFacade(theput, mcengine, thedata)
price2, stderr2 = option2.price()
print("The put price via Delta Control Variate Monte Carlo is: {0:.4f} and the standard error is {1:.4f}.".format(price2, stderr2))



