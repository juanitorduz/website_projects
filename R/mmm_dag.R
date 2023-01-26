library(dagitty)

dag <- dagitty( x = "dag {
  TV -> Search -> Conversions
  TV -> Conversions
  Seasonality -> Conversions
  Seasonality -> Search
  Seasonality -> Facebook -> Search
}" )

coordinates( x = dag ) <- list(
  x = c(
    "Seasonality" = -1,
    "TV" = 0,
    "Search" = 1,
    "Facebook" = 0,
    "Conversions" = 2
  ),
  y = c(
    "Seasonality" = 1,
    "TV" = -1,
    "Search" = 0,
    "Facebook" = 0,
    "Conversions" = 0
  )
)

plot( dag )
