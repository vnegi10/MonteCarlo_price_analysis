## MonteCarlo_price_analysis

In this Pluto notebook, we will predict prices of various assets (stocks, cryptos etc.) by
using Monte Carlo simulations. Historical price data is obtained via the Alpha Vantage API,
which is implemented using the [AlphaVantage.jl](https://github.com/ellisvalentiner/AlphaVantage.jl) package.

## API key

You will need to create an API key. It can be done for 
free [here.](https://www.alphavantage.co/support/#api-key)

## How to use?

Install Pluto.jl (if not done already) by executing the following commands in your Julia REPL:

    using Pkg
    Pkg.add("Pluto")
    using Pluto
    Pluto.run() 

Clone this repository and open **MonteCarlo_notebook.jl** in your Pluto browser window. That's it!
You are good to go.