# Load necessary libraries
library(tidyverse)
library(lubridate)
library(forecast)
library(tseries)
library(vars)
library(astsa)
library(ggplot2)
library(dplyr)

# Load Data
df <- read.csv("https://dxl-datasets.s3.us-east-1.amazonaws.com/data/industry_data_train.csv")

# Convert to Date format
df <- df %>%
  mutate(date = as.Date(paste(date, "-01", sep=""), format="%Y-%m-%d"))

# Visualizing the data
df %>%
  pivot_longer(cols = -date, names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = date, y = Value, color = Variable)) +
  geom_line() +
  theme_minimal() +
  labs(title = "Retail Industry Time Series Data", x = "Month", y = "Value")


# fit ts model and split train and test data
jobs_ts <- ts(df$jobs, start = c(2003,1), frequency = 12)
employ_ts <- ts(df$employees, start = c(2003,1), frequency = 12)
layoffs_ts <- ts(df$layoffs, start = c(2003,1), frequency = 12)
spend_ts <- ts(df$spend, start = c(2003,1), frequency = 12)
train1 <- window(jobs_ts, end = c(2013,1)) 
train2 <- window(employ_ts, end = c(2013,1))
train4 <- window(spend_ts, end = c(2013,1))
test1 <- window(jobs_ts, start = c(2013,2), end = end(jobs_ts)) 
test2 <- window(employ_ts, start = c(2013,2))
test4 <- window(spend_ts, start = c(2013,2))
# Check Stationarity for train1
adf.test(diff(train1))
acf(diff(train1, lag=12))  # Seasonal ACF
pacf(diff(train1, lag=12)) # Seasonal PACF

seas_decomp <- stl(train1, s.window="periodic")
plot(seas_decomp)
nsdiffs(train1)

#ARIMA(1,1,0)
sarima(train1, p=1, d=1, q=0)
#ARIMA(1,1,1)
sarima(train1, p=1, d=1, q=1)
#ARIMA(0,1,1)
sarima(train1, p=0, d=1, q=1)

# Fit the best ARIMA Model 
m_arima <- Arima(train1, order=c(1,1,1),seasonal = c(1,1,0))

# Fit ARIMAX Model
# Fit ARIMAX with multiple regressors
train_xreg <- cbind(train2, train4)

m_x <- tslm(train1 ~ train_xreg) 
adf.test(diff(train2))
adf.test(diff(train4))

r <- rstudent(m_x)
acf(r,lag = 12)
pacf(r, lag = 12)
sarima(r, 1, 1, 2)
m_arimax <- Arima(train1, order = c(1,1,2), seasonal = c(1,1,1), xreg = train_xreg,method = 'CSS')
# Fit Vector Autoregression (VAR) 
us_data = cbind(jobs_ts,employ_ts,layoffs_ts,spend_ts)
us_data_train = window(us_data, end=c(2013, 1))
# Convert to time series format
us_ts <- ts(us_data_train, start = c(2003,1), frequency = 12)
VARselect(us_data)

m_var <- VAR(us_ts, p = 10)

# Model Comparsion
f_arima <- forecast(m_arima, h=length(test1))
f_arimax <- forecast(m_arimax, xreg=train_xreg)
f_var <- forecast(m_var, h=length(test1))
# Plot Forecast
autoplot(test1, series="Test Data") +
  autolayer(f_arima$mean, series="ARIMA(1,1,1) Forecast") +
  autolayer(f_arimax$mean, series="ARIMAX(1,0,0) Forecast") +
  autolayer(f_var$forecast$jobs, series="VAR(10) Forecast", PI=FALSE) +
  ggtitle("Forecast Comparison: ARIMA vs ARIMAX vs VAR") +
  ylab("Value") +
  theme_minimal()

rmse_arima <- accuracy(f_arima, test1)[2,"RMSE"]  
rmse_arimax <- accuracy(f_arimax, test1)[2,"RMSE"]  
rmse_var <- accuracy(f_var$forecast$jobs, test1)[2,"RMSE"]  

#Since rmse_arima is the smallest one, we choose Arimax model as our final model.
test_xreg <- cbind(test2, test4)
final_model <- Arima(test1, order = c(1,1,2), seasonal = c(1,1,1), xreg = test_xreg ,method = 'CSS')
forecast_arimax <- forecast(final_model, xreg = test_xreg)
forecast_2024 <- window(forecast_arima$mean, start = c(2024, 1), end = c(2024, 12))

# Convert to DataFrame
forecast_2024_df<- as.data.frame(forecast_2024)
forecast_dates <- seq(as.Date("2024-01-01"), by = "month", length.out = 12)

# Create a forecast dataframe
forecast_df <- data.frame(
  Month = forecast_dates,
  Forecast = forecast_2024_df$x
)


# Save to CSV
write_csv(forecast_df, "Final_Forecast.csv")


