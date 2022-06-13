### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ 1cddaf80-ea84-11ec-133c-7dfc20345e12
using AlphaVantage, DataFrames, Dates, Statistics, VegaLite, Distributions, JSON

# ╔═╡ 2ff8e8e5-2884-442f-acce-1555f45c100f
md"
### Load packages
---
"

# ╔═╡ 687a46a9-0305-497c-9e9a-522af8b29a53
md"
### API key
---
"

# ╔═╡ 64fa2b95-efa5-4110-a476-fe51fb49bb63
begin
	client = AlphaVantage.GLOBAL[]
	key_dict = ""
	if Sys.iswindows()
		key_dict = JSON.parsefile(raw"C:\Users\vnegi\Documents\VNEG_alphavantage_key.json.txt")
	else
		key_dict = JSON.parsefile("/home/vikas/Documents/Input_JSON/VNEG_alphavantage_key.json")
	end
	client.key = key_dict["api_key"]
end

# ╔═╡ 84ae984a-5e7c-4b58-8c2b-717613c731db
md"
### Get stocks data
---
"

# ╔═╡ 0c1d14ac-4ca3-4b01-8c03-7219b007e10b
function raw_to_df(raw_data)

    # Argument ":auto" is required to generate column names in latest version of DataFrames.jl (v1.2.1)    
    df = DataFrame(raw_data[1], :auto) 
    df_names = Symbol.(vcat(raw_data[2]...))
    df = DataFrames.rename(df, df_names)

    timestamps = df[!,:timestamp]

    select!(df, Not([:timestamp]))

    for col in eachcol(df)
        col = Float64.(col)
    end

    df[!,:Date] = Date.(timestamps)
	
    return sort(df, :Date)
end

# ╔═╡ 9173078f-e257-4c75-be84-a8c9e5a04502
md"
##### % change for closing prices
"

# ╔═╡ 04a2d403-39b0-433e-ad0e-b162f352bad6
function get_stock_change(stock_name::String, duration::Int64)

	raw_data = time_series_daily(stock_name, datatype = "csv", outputsize = "full")
	df_raw = raw_to_df(raw_data)

	rows, cols = size(df_raw)
	@assert duration < rows "Input duration is larger than available data"

	df_duration = df_raw[end-duration:end, :]

	change = ((df_duration[!, :close][2:end] .- 
               df_duration[!, :close][1:end-1]) ./ 
	           df_duration[!, :close][1:end-1]) * 100

	μ = Statistics.mean(change)
	σ = Statistics.std(change)

	return select(df_duration, [:Date, :close]), μ, σ
end	

# ╔═╡ a17a3aea-411e-4acf-8f81-8bd11b742967
function get_stock_change_norm(stock_name::String, duration::Int64)

	raw_data = time_series_daily(stock_name, datatype = "csv", outputsize = "full")
	df_raw = raw_to_df(raw_data)

	rows, cols = size(df_raw)
	@assert duration < rows "Input duration is larger than available data"

	df_duration = df_raw[end-duration:end, :]

	change = ((df_duration[!, :close][2:end] .- 
               df_duration[!, :close][1:end-1]) ./ 
	           df_duration[!, :close][1:end-1])

	μ = Statistics.mean(change)
	σ = Statistics.std(change)

	return select(df_duration, [:Date, :close]), μ, σ
end	

# ╔═╡ 0b227807-e7ee-417f-9c2b-96e281aee639
results = get_stock_change("AAPL", 10)

# ╔═╡ 3e35912d-a7ba-4411-b7f8-96cfc5d9e3a3
df_close, μ, σ = results[1], results[2], results[3];

# ╔═╡ 15d0e6aa-e324-4725-955c-fdefff0d81ba
md"
### Perform MC simulations
---
"

# ╔═╡ 19a5209f-2693-42a3-8c3a-10201c6c551a
function run_mc_simulation(df_close::DataFrame, num_sim::Int64, 
	                       days_to_sim::Int64, mean::Float64,
                           std_dev::Float64)
	
	df_predict  = DataFrame()	
	new_date    = df_close[!, :Date][end]	

	# Generate future dates only once
	dates = [df_close[!, :Date][end]]
	for j = 1:days_to_sim

		new_date += Dates.Day(1)
		
		# Prices are not reported for weekends
		if Dates.dayname(new_date) == "Saturday"
			new_date = new_date + Dates.Day(2)
		elseif Dates.dayname(new_date) == "Sunday"
			new_date = new_date + Dates.Day(1)
		end

		push!(dates, new_date)
				
	end	
	
	for i = 1:num_sim	

		close_price = [df_close[!, :close][end]]	    
		
		for j = 1:days_to_sim
			change_in_percentage = rand(Normal(mean, std_dev), 1)[1]
			new_close_price = close_price[end] * (1 + change_in_percentage/100)
			push!(close_price, new_close_price)
		end

		if isempty(df_predict)
			df_predict = DataFrame("Date" => dates, "close_$(i)" => close_price)
		else
			df_to_join = DataFrame("Date" => dates, "close_$(i)" => close_price)
			df_predict = innerjoin(df_predict, df_to_join, on = :Date)
		end
		
	end

	return df_predict	
end

# ╔═╡ 242433de-5a81-4f3b-87f3-40efc7ac2056
function run_mc_simulation_exp(df_close::DataFrame, num_sim::Int64, 
	                           days_to_sim::Int64, mean::Float64,
                               std_dev::Float64)
	
	df_predict  = DataFrame()
	last_date   = df_close[!, :Date][end]

	# Generate future dates only once
	dates = [df_close[!, :Date][end]]
	for j = 1:days_to_sim
		new_date = last_date + Dates.Day(j)
		push!(dates, new_date)
	end	
	
	for i = 1:num_sim	

		close_price = [df_close[!, :close][end]]
	   		
		for j = 1:days_to_sim
			change_in_norm = rand(Normal(mean, std_dev), 1)[1]

			# Exponential growth
			new_close_price = close_price[end] * exp(change_in_norm)
			push!(close_price, new_close_price)
		end

		if isempty(df_predict)
			df_predict = DataFrame("Date" => dates, "close_$(i)" => close_price)
		else
			df_to_join = DataFrame("Date" => dates, "close_$(i)" => close_price)
			df_predict = innerjoin(df_predict, df_to_join, on = :Date)
		end
		
	end

	return df_predict	
end

# ╔═╡ 686340c1-15a8-4d3a-91b6-3e4020a7e93f
df_predict = run_mc_simulation(df_close, 5, 11, μ, σ)

# ╔═╡ effbb61d-2a29-4909-80e3-caf0626daee3
function get_mc_avg(df_predict::DataFrame)

	close_avg = Float64[]
	for row in eachrow(df_predict)
		avg_price = Statistics.mean(row[2:end])
		push!(close_avg, avg_price)
	end

	return DataFrame(Date = df_predict[!, :Date], close_avg = close_avg)
end	

# ╔═╡ fc9d3eb1-5344-46a1-9d7b-b9076449353c


# ╔═╡ 6f15d7a5-b565-4bbd-948b-b02d7d489d3b
md"
### Plot prediction data
---
"

# ╔═╡ ee9081ad-132f-4d85-9c69-0a6a26594e91
function plot_mc_prediction(stock_name::String; duration::Int64 = 180,
                            num_sim::Int64 = 200, days_to_sim::Int64 = 30)

	results = get_stock_change(stock_name, duration)
	df_close, μ, σ = results[1], results[2], results[3]	

	# Get predictions from MC simulations
	df_predict = run_mc_simulation(df_close, num_sim, days_to_sim, μ, σ)

	# Get average price for all MC simulations
	df_avg = get_mc_avg(df_predict)
	
	predicted_closing_price = df_avg[!, :close_avg][end]
	known_closing_price     = df_close[!, :close][end]

	predicted_change = ((predicted_closing_price - 
	                     known_closing_price)/known_closing_price) * 100

	# Add average price column to DataFrame with predicted prices
	insertcols!(df_predict, :Date, :close_avg => df_avg[!, :close_avg], after = true)
	
	sdf_predict = stack(df_predict, Not([:Date, :close_avg]), 
		                variable_name = :sim_number)

	figure = sdf_predict |>

	@vlplot(:line, 
	        x = {:Date, "axis" = {"title" = "Time [days]", "labelFontSize" = 12, "titleFontSize" = 14}, "type" = "temporal"},
	        width = 750, height = 500, 
			"title" = {"text" = "$(stock_name) predicted price is $(round(predicted_closing_price, digits = 2)), hist. duration = $(duration) days, change = $(round(predicted_change, digits = 2)) %, calculated after $(days_to_sim) days", "fontSize" = 16},
			) +

	@vlplot(:line, 
	        y = {:value, "axis" = {"title" = "Predicted price [USD]", 
			"labelFontSize" = 12, "titleFontSize" = 14}}, 
	        color = :sim_number) +

	@vlplot(mark = {:line, point = false, strokeWidth = 5},	        
	        y = {:close_avg, "scale" = {"zero" = true}})	

	return figure
end	

# ╔═╡ 087b7abe-df05-4e63-9d3b-4289f58d9c4a
function plot_mc_prediction_exp(stock_name::String; duration::Int64 = 180,
                                num_sim::Int64 = 200, days_to_sim::Int64 = 30)

	results = get_stock_change_norm(stock_name, duration)
	df_close, μ, σ = results[1], results[2], results[3]		

	df_predict = run_mc_simulation_exp(df_close, num_sim, days_to_sim, μ, σ)

	predicted_closing_price = sum(df_predict[end, :][2:end])/num_sim
	known_closing_price     = df_close[!, :close][end]

	predicted_change = ((predicted_closing_price - 
	                     known_closing_price)/known_closing_price) * 100
	
	sdf_predict = stack(df_predict, Not([:Date]), variable_name = :sim_number)

	figure = sdf_predict |>

	@vlplot(:line, 
	        x = {:Date, "axis" = {"title" = "Time [days]", "labelFontSize" = 12, "titleFontSize" = 14}, "type" = "temporal"},
	        y = {:value, "axis" = {"title" = "Price [USD]", "labelFontSize" = 12, "titleFontSize" = 14}},
	        width = 750, height = 500, 
			"title" = {"text" = "$(stock_name) avg. predicted price is $(round(predicted_closing_price, digits = 2)) with a change of $(round(predicted_change, digits = 2)) %, calculated after $(days_to_sim) days from exp growth" , "fontSize" = 16},
			color = :sim_number)

	return figure
end	

# ╔═╡ 8398a7ec-650c-4d1a-a63f-9d70d0ae6355
plot_mc_prediction("MSFT", duration = 5*365, num_sim = 100, days_to_sim = 365)

# ╔═╡ aa6d8740-339f-40ac-a3b9-59fc3d754c72
#plot_mc_prediction("MSFT", duration = 2*365, num_sim = 1000, days_to_sim = 90)

# ╔═╡ 8d538a46-ae6c-433f-b684-39f3e4c2839f
plot_mc_prediction_exp("MSFT", duration = 1*365, num_sim = 1000, days_to_sim = 90)

# ╔═╡ 610ea9a1-f73d-4723-ad10-5673894da725
#plot_mc_prediction_exp("MSFT", duration = 2*365, num_sim = 1000, days_to_sim = 90)

# ╔═╡ 4c10f658-64f9-4c18-9a73-8bd6ad5a44ff
#plot_mc_prediction("AAPL", duration = 1*365, num_sim = 1000, days_to_sim = 180)

# ╔═╡ 0a98ae67-26c7-4bdb-af24-07be0d222c0a
#plot_mc_prediction("AMD", duration = 1*365, num_sim = 1000, days_to_sim = 180)

# ╔═╡ 4fc4819f-5994-479d-b01f-bf1acb4d4fb9
#plot_mc_prediction("COIN", duration = 250, num_sim = 1000, days_to_sim = 180)

# ╔═╡ dcb81a30-b89b-4f4e-a49e-7132d9489dad
md"
### Plot final price distribution
---
"

# ╔═╡ 415a1ca7-8227-4275-a7c4-1fdbfda7b071
function plot_mc_distribution(stock_name::String; duration::Int64 = 180,
                            num_sim::Int64 = 200, days_to_sim::Int64 = 30)

	results = get_stock_change(stock_name, duration)
	df_close, μ, σ = results[1], results[2], results[3]		

	df_predict = run_mc_simulation(df_close, num_sim, days_to_sim, μ, σ)

	predicted_closing_price = sum(df_predict[end, :][2:end])/num_sim
	known_closing_price     = df_close[!, :close][end]

	predicted_change = ((predicted_closing_price - 
	                       known_closing_price)/known_closing_price) * 100

	# Collect prices from last row for plotting histogram
	df_last_row = df_predict[end, 2:end]
	final_prices = Float64[]
	
	for i = 1:num_sim
		push!(final_prices, df_last_row[i])
	end

	df_final_price = DataFrame(sim_number = names(df_last_row), 
		                       final_prices = final_prices)	

	figure = df_final_price |>

	@vlplot(:bar, 
	        x = {:final_prices, "axis" = {"title" = "Final predicted price [USD]", "labelFontSize" = 12, "titleFontSize" = 14}, "bin" = {"maxbins" = 50}},
	        y = {"count()", "axis" = {"title" = "Number of counts", "labelFontSize" = 12, "titleFontSize" = 14}},
	        width = 750, height = 500, 
			"title" = {"text" = "$(stock_name) distribution, hist. duration = $(duration) days, avg. predicted price = $(round(predicted_closing_price, digits = 2)), change = $(round(predicted_change, digits = 2)) %, calculated after $(days_to_sim) days", "fontSize" = 16})

	return figure
end	

# ╔═╡ c4e0fb37-3010-4545-ad2f-49128c22b27d
#plot_mc_distribution("MSFT", duration = 1*365, num_sim = 10_000, days_to_sim = 90)

# ╔═╡ f87be5e1-c992-48bd-81d1-872d17514886
#plot_mc_distribution("MSFT", duration = 2*365, num_sim = 10_000, days_to_sim = 90)

# ╔═╡ e635c722-bda3-467f-bebe-a3a97a5e0326
#plot_mc_distribution("MSFT", duration = 5*365, num_sim = 10_000, days_to_sim = 90)

# ╔═╡ 8e6a9a18-a1c5-42b8-a5b6-0ee66d79b3aa
md"
### Backtesting MC prediction
---
"

# ╔═╡ 4c2878ee-5a0b-4f67-aebd-ecd77919e200
function get_stock_change_bt(stock_name::String, duration::Int64,
	                         backtest::Int64)

	raw_data = time_series_daily(stock_name, datatype = "csv", outputsize = "full")
	df_raw = raw_to_df(raw_data)

	rows, cols = size(df_raw)
	@assert duration + backtest < rows "Input duration + backtest is larger than available data"

	df_duration = df_raw[end-backtest-duration:end-backtest, :]

	change = ((df_duration[!, :close][2:end] .- 
               df_duration[!, :close][1:end-1]) ./ 
	           df_duration[!, :close][1:end-1]) * 100

	μ = Statistics.mean(change)
	σ = Statistics.std(change)

	df_close_bt = select(df_duration, [:Date, :close])
	df_raw_bt   = select(df_raw, [:Date, :close]) 

	return df_close_bt, df_raw_bt, μ, σ
end	

# ╔═╡ 5d27d50e-1b8b-450f-988f-bb94153e3b4d
function run_mc_simulation_bt(df_close::DataFrame, df_raw::DataFrame,
	                          num_sim::Int64, days_to_sim::Int64, mean::Float64,
                              std_dev::Float64)
	
	df_predict  = DataFrame()	
	
	dates = df_raw[!, :Date][end-days_to_sim:end]	
	
	for i = 1:num_sim	

		close_price = [df_close[!, :close][end]]	    
		
		for j = 1:days_to_sim
			change_in_percentage = rand(Normal(mean, std_dev), 1)[1]
			new_close_price = close_price[end] * (1 + change_in_percentage/100)
			push!(close_price, new_close_price)
		end

		if isempty(df_predict)
			df_predict = DataFrame("Date" => dates, "close_$(i)" => close_price)
		else
			df_to_join = DataFrame("Date" => dates, "close_$(i)" => close_price)
			df_predict = innerjoin(df_predict, df_to_join, on = :Date)
		end
		
	end

	insertcols!(df_predict, :Date, 
		        :close_actual => Float64.(df_raw[!, :close][end-days_to_sim:end]),
	            after = true)

	return df_predict	
end

# ╔═╡ 48c0a535-8163-46a4-bb1f-58ce5d67aeb0
function get_mc_avg_bt(df_predict::DataFrame)

	close_avg = Float64[]
	for row in eachrow(df_predict)
		avg_price = Statistics.mean(row[3:end])
		push!(close_avg, avg_price)
	end

	df_avg_bt = DataFrame(Date = df_predict[!, :Date], 
		                  close_avg = close_avg,
	                      close_actual = df_predict[!, :close_actual])

	return df_avg_bt
end	

# ╔═╡ 5343d494-caa6-4dbf-aa3f-3aaccf3b4013
df_close_bt = get_stock_change_bt("AAPL", 20, 5)[1];

# ╔═╡ 30dde1da-33aa-4b3e-a4ce-48c158238615
df_raw_bt = get_stock_change_bt("AAPL", 20, 5)[2];

# ╔═╡ 26c13a9d-d378-4101-b5d0-ba1d5f65ab5b
df_predict_bt = run_mc_simulation_bt(df_close_bt, df_raw_bt, 5, 5, μ, σ)

# ╔═╡ 00b556e6-015c-4d0f-9d36-8e9de84c824c
df_avg_bt = get_mc_avg_bt(df_predict_bt);

# ╔═╡ 9548e8b3-460d-4e3c-9ba6-11ae7ab2d066
function plot_mc_prediction_bt(stock_name::String; duration::Int64 = 180,
                               backtest::Int64 = 30, num_sim::Int64 = 200)

	results = get_stock_change_bt(stock_name, duration, backtest)
	df_close_bt, df_raw_bt, μ, σ = results[1], results[2], results[3], results[4]	

	df_predict_bt = run_mc_simulation_bt(df_close_bt, df_raw_bt, 
		                                 num_sim, backtest, μ, σ)

	df_avg_bt = get_mc_avg_bt(df_predict_bt)
	
	sdf_predict = stack(df_avg_bt, Not([:Date]), variable_name = :close_data)

	figure = sdf_predict |>

	@vlplot(mark = {:line, point = true}, 
	        x = {:Date, "axis" = {"title" = "Time [days]", "labelFontSize" = 12, "titleFontSize" = 14}, "type" = "temporal"},
	        y = {:value, "axis" = {"title" = "Price [USD]", "labelFontSize" = 12, "titleFontSize" = 14}, "type" = "quantitative", "scale" = {"zero" = false}},
	        width = 750, height = 500, 
			"title" = {"text" = "$(stock_name) avg. predicted price vs actual price backtested for $(backtest) days" , "fontSize" = 16},
			color = :close_data)

	return figure
end	

# ╔═╡ a6a8ae09-87cd-4f93-8ea7-fb10c92270bb
plot_mc_prediction_bt("AAPL", duration = 5*365, backtest = 365, num_sim = 2000)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AlphaVantage = "6348297c-a006-11e8-3a05-9bbf8830fd7b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
VegaLite = "112f6efa-9a02-5b7d-90c0-432ed331239a"

[compat]
AlphaVantage = "~0.4.1"
DataFrames = "~1.3.4"
Distributions = "~0.25.62"
JSON = "~0.21.3"
VegaLite = "~2.6.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AlphaVantage]]
deps = ["ArgCheck", "Compat", "DelimitedFiles", "HTTP", "HttpCommon", "JSON", "Tables"]
git-tree-sha1 = "893a38118dc5a7554a52bbbc58d6de591e58f7b6"
uuid = "6348297c-a006-11e8-3a05-9bbf8830fd7b"
version = "0.4.1"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9489214b993cd42d17f44c36e359bf6a7c919abf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "0ec161f87bf4ab164ff96dfacf4be8ffff2375fd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.62"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "9267e5f50b0e12fdfd5a2455534345c4cf2c7f7a"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.14.0"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HttpCommon]]
deps = ["Dates", "Nullables", "Test", "URIParser"]
git-tree-sha1 = "46313284237aa6ca67a6bce6d6fbd323d19cff59"
uuid = "77172c1b-203f-54ac-aa54-3f1198fe9f90"
version = "0.5.0"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "cb7099a0109939f16a4d3b572ba8396b1f6c7c31"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.10"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JSONSchema]]
deps = ["HTTP", "JSON", "URIs"]
git-tree-sha1 = "2f49f7f86762a0fbbeef84912265a1ae61c4ef80"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "0.3.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.NodeJS]]
deps = ["Pkg"]
git-tree-sha1 = "905224bbdd4b555c69bb964514cfa387616f0d3a"
uuid = "2bd173c7-0d6d-553b-b6af-13a54713934c"
version = "1.3.0"

[[deps.Nullables]]
git-tree-sha1 = "8f87854cc8f3685a60689d8edecaa29d2251979b"
uuid = "4d1e1d77-625e-5b40-9113-a560ec7a8ecd"
version = "1.0.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "3e32c8dbbbe1159a5057c80b8a463369a78dd8d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.12"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "fca29e68c5062722b5b4435594c3d1ba557072a3"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.7.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "a9e798cae4867e3a41cae2dd9eb60c047f1212db"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.6"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2c11d7290036fe7aac9038ff312d3b3a2a5bf89e"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.4.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.TableTraitsUtils]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Missings", "TableTraits"]
git-tree-sha1 = "78fecfe140d7abb480b53a44f3f85b6aa373c293"
uuid = "382cd787-c1b6-5bf2-a167-d5b971a19bda"
version = "1.0.2"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Vega]]
deps = ["DataStructures", "DataValues", "Dates", "FileIO", "FilePaths", "IteratorInterfaceExtensions", "JSON", "JSONSchema", "MacroTools", "NodeJS", "Pkg", "REPL", "Random", "Setfield", "TableTraits", "TableTraitsUtils", "URIParser"]
git-tree-sha1 = "43f83d3119a868874d18da6bca0f4b5b6aae53f7"
uuid = "239c3e63-733f-47ad-beb7-a12fde22c578"
version = "2.3.0"

[[deps.VegaLite]]
deps = ["Base64", "DataStructures", "DataValues", "Dates", "FileIO", "FilePaths", "IteratorInterfaceExtensions", "JSON", "MacroTools", "NodeJS", "Pkg", "REPL", "Random", "TableTraits", "TableTraitsUtils", "URIParser", "Vega"]
git-tree-sha1 = "3e23f28af36da21bfb4acef08b144f92ad205660"
uuid = "112f6efa-9a02-5b7d-90c0-432ed331239a"
version = "2.6.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─2ff8e8e5-2884-442f-acce-1555f45c100f
# ╠═1cddaf80-ea84-11ec-133c-7dfc20345e12
# ╟─687a46a9-0305-497c-9e9a-522af8b29a53
# ╟─64fa2b95-efa5-4110-a476-fe51fb49bb63
# ╟─84ae984a-5e7c-4b58-8c2b-717613c731db
# ╟─0c1d14ac-4ca3-4b01-8c03-7219b007e10b
# ╟─9173078f-e257-4c75-be84-a8c9e5a04502
# ╟─04a2d403-39b0-433e-ad0e-b162f352bad6
# ╟─a17a3aea-411e-4acf-8f81-8bd11b742967
# ╠═0b227807-e7ee-417f-9c2b-96e281aee639
# ╠═3e35912d-a7ba-4411-b7f8-96cfc5d9e3a3
# ╟─15d0e6aa-e324-4725-955c-fdefff0d81ba
# ╟─19a5209f-2693-42a3-8c3a-10201c6c551a
# ╟─242433de-5a81-4f3b-87f3-40efc7ac2056
# ╠═686340c1-15a8-4d3a-91b6-3e4020a7e93f
# ╟─effbb61d-2a29-4909-80e3-caf0626daee3
# ╠═fc9d3eb1-5344-46a1-9d7b-b9076449353c
# ╟─6f15d7a5-b565-4bbd-948b-b02d7d489d3b
# ╟─ee9081ad-132f-4d85-9c69-0a6a26594e91
# ╟─087b7abe-df05-4e63-9d3b-4289f58d9c4a
# ╠═8398a7ec-650c-4d1a-a63f-9d70d0ae6355
# ╠═aa6d8740-339f-40ac-a3b9-59fc3d754c72
# ╠═8d538a46-ae6c-433f-b684-39f3e4c2839f
# ╠═610ea9a1-f73d-4723-ad10-5673894da725
# ╠═4c10f658-64f9-4c18-9a73-8bd6ad5a44ff
# ╠═0a98ae67-26c7-4bdb-af24-07be0d222c0a
# ╠═4fc4819f-5994-479d-b01f-bf1acb4d4fb9
# ╟─dcb81a30-b89b-4f4e-a49e-7132d9489dad
# ╟─415a1ca7-8227-4275-a7c4-1fdbfda7b071
# ╠═c4e0fb37-3010-4545-ad2f-49128c22b27d
# ╠═f87be5e1-c992-48bd-81d1-872d17514886
# ╠═e635c722-bda3-467f-bebe-a3a97a5e0326
# ╟─8e6a9a18-a1c5-42b8-a5b6-0ee66d79b3aa
# ╟─4c2878ee-5a0b-4f67-aebd-ecd77919e200
# ╟─5d27d50e-1b8b-450f-988f-bb94153e3b4d
# ╟─48c0a535-8163-46a4-bb1f-58ce5d67aeb0
# ╠═5343d494-caa6-4dbf-aa3f-3aaccf3b4013
# ╠═30dde1da-33aa-4b3e-a4ce-48c158238615
# ╠═26c13a9d-d378-4101-b5d0-ba1d5f65ab5b
# ╠═00b556e6-015c-4d0f-9d36-8e9de84c824c
# ╟─9548e8b3-460d-4e3c-9ba6-11ae7ab2d066
# ╠═a6a8ae09-87cd-4f93-8ea7-fb10c92270bb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
