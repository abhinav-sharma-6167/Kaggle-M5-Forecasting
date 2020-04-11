#-----------------------------------------------------------------------------------
setwd("C:/Users/as16220/Desktop/Kaggle/M5")
suppressMessages({
  library(data.table)
  library(RcppRoll)
  library(dplyr)
  library(vroom)
})

# Garbage collection
igc <- function() {
  invisible(gc()); invisible(gc())   
}

#-----------------------------------------------------------------------------------
path <- "C:/Users/as16220/Desktop/Kaggle/M5/"

calendar <- as.data.table(vroom::vroom(file.path(path, "calendar.csv")))
selling_prices <- as.data.table(vroom::vroom(file.path(path, "sell_prices.csv")))
sample_submission <- as.data.table(vroom::vroom(file.path(path, "sample_submission.csv")))
sales <- as.data.table(vroom::vroom(file.path(path, "sales_train_validation.csv")))

#-----------------------------------------------------------------------------------
calendar[, `:=`(date = NULL, 
                weekday = NULL, 
                d = as.integer(substring(d, 3)))]
cols <- c("event_name_1", "event_type_1") #, "event_name_2", "event_type_2")
calendar[, (cols) := lapply(.SD, function(z) as.integer(as.factor(z))), .SDcols = cols]

# Selling prices
selling_prices[, `:=`(
  sell_price_rel_diff = sell_price / dplyr::lag(sell_price) - 1,
  sell_price_cumrel = (sell_price - cummin(sell_price)) / (1 + cummax(sell_price) - cummin(sell_price)),
  sell_price_roll_sd7 = roll_sdr(sell_price, n = 7)
), by = c("store_id", "item_id")]

# Sales: Reshape
sales[, id := gsub("_validation", "", id)]                         
empty_dt = matrix(NA_integer_, ncol = 2 * 28, nrow = 1, dimnames = 
                    list(NULL, paste("d", 1913 + 1:(2 * 28), sep = "_")))

sales <- cbind(sales, empty_dt)
sales <- melt(sales, id.vars = c("id", "item_id", "dept_id", "cat_id", "store_id", "state_id"), 
              variable.name = "d", value = "demand")
sales[, d := as.integer(substring(d, 3))]

# Sales: Reduce size
sales <- sales[d > 1000]

# Sales: Feature construction
stopifnot(!is.unsorted(sales$d))
sales[, lag_t28 := dplyr::lag(demand, 28), by = "id"]
sales[, `:=`(rolling_mean_t7 = roll_meanr(lag_t28, 7),
             rolling_mean_t30 = roll_meanr(lag_t28, 30),
             rolling_mean_t60 = roll_meanr(lag_t28, 60),
             rolling_mean_t90 = roll_meanr(lag_t28, 90),
             rolling_mean_t180 = roll_meanr(lag_t28, 180),
             rolling_sd_t7 = roll_sdr(lag_t28, 7),
             rolling_sd_t30 = roll_sdr(lag_t28, 30)), 
      by = "id"]
igc()

# Reduce size again
sales <- sales[(d >= 1800) | !is.na(rolling_mean_t180)]

sales <- calendar[sales, on = "d"]
igc()

# Merge selling prices to sales and drop key
train <- selling_prices[sales, on = c('store_id', 'item_id', 'wm_yr_wk')][, wm_yr_wk := NULL]
rm(sales, selling_prices, calendar)
igc()

#-----------------------------------------------------------------------------------


# Integer coding of categoricals, scaling and imputation for numerics
cat_cols = c("item_id", "wday", "month", "year", 
             "event_name_1", "event_type_1", "event_name_2", "event_type_2", 
             "dept_id", "store_id", "cat_id", "state_id")

dense_cols = c("sell_price", "sell_price_rel_diff", "sell_price_cumrel", "sell_price_roll_sd7",
               "lag_t28", "rolling_mean_t7", "rolling_mean_t30", "rolling_mean_t60", "rolling_mean_t90", 
               "rolling_mean_t180", "rolling_sd_t7", "rolling_sd_t30",
               "snap_CA", "snap_TX", "snap_WI")

train[, (cat_cols) := lapply(.SD, function(z) as.integer(as.factor(z))), .SDcols = cat_cols]
igc()
train[, (dense_cols) := lapply(.SD, function(z) {z[is.na(z)] <- mean(z, na.rm = TRUE); z}), 
      .SDcols = dense_cols]
igc()

#-----------------------------------------------------------------------------------
# Separate submission data and reconstruct id columns
test <- train[train$d >= 1914]
test[, id := paste(id, ifelse(d <= 1941, "validation", "evaluation"), sep = "_")]
test[, F := paste0("F", d - 1913 - 28 * (d > 1941))]

# 1 month of validation data
flag <- train$d < 1914 & train$d >= 1914 - 28
valid <- list(train[flag], train[["demand"]][flag])

# Training data
flag <- train$d < 1914 - 28
y <- train[["demand"]][flag]
X <- train[flag]
rm(train, flag)
igc()

#-----------------------------------------------------------------------------------
#DL
library(keras)
#library(tensorflow)
library(reticulate)
#library(kerasR)
library(mlr)

# X_id <- X$id
# X[,id := NULL]
# test_id <- test$id
# test_F <- test$`F`
# test[,id:=NULL];test[,`F`:=NULL]
# valid_id <- valid[[1]]$id
# valid[[1]][,id := NULL]
# 
# X[,demand := NULL]
# y_hat <- test$demand
# test[,demand:=NULL]
# valid[[1]][,demand := NULL]
# 
# X <- data.matrix(X)
# y <- array(y)
# test <- data.matrix(test)
# y_hat <- array(y_hat)
# valid[[1]] <- data.matrix(valid[[1]])
# valid[[2]] <- array(valid[[2]])
# names_vec <- dimnames(X)
# dimnames(X) <- NULL
# dimnames(test) <- NULL
# dimnames(valid[[1]]) <- NULL





#-----------------------------------------------------------------------------------
#tensorflow::install_tensorflow(method = "conda")
#keras::install_keras(method = "conda")

#Define the structure
#X <- keras::normalize(X) ; test <- keras::normalize(test) ; valid[[1]] <- keras::normalize(valid[[1]])


# library(clusterSim)
# data_dt <- as.data.table(rbind(X,valid[[1]],test))
# data_dt <- clusterSim::data.Normalization(data_dt,type="n1",normalization="column")
# X <-data.matrix(data_dt[1:dim(X)[[1]]])
# valid[[1]] <- data.matrix()



library(keras)

# DEFINE MODEL
new_model <- function(lr = 0.00007) {
  k_clear_session()
  set.seed(92)
  #tensorflow::tf$random$set_random_seed(83)
  
  # Dense part
  # dense_input = layer_input(length(dense_cols), name = "dense1", dtype = "float32")
  # 
  # Embedded layer_input
  wday_input = layer_input(1, name = "wday", dtype = "int8")
  month_input = layer_input(1, name = "month", dtype = "int8")
  year_input = layer_input(1, name = "year", dtype = "int8")
  event_name_1_input = layer_input(1, name = "event_name_1", dtype = "int8")
  event_type_1_input = layer_input(1, name = "event_type_1", dtype = "int8")
  event_name_2_input = layer_input(1, name = "event_name_2", dtype = "int8")
  event_type_2_input = layer_input(1, name = "event_type_2", dtype = "int8")
  item_id_input = layer_input(1, name = "item_id", dtype = "int16")
  dept_id_input = layer_input(1, name = "dept_id", dtype = "int8")
  store_id_input = layer_input(1, name = "store_id", dtype = "int8")
  cat_id_input = layer_input(1, name = "cat_id", dtype = "int8")
  state_id_input = layer_input(1, name = "state_id", dtype = "int8")

  # embedding layers
  wday_emb = wday_input %>% 
    layer_embedding(8, 1) %>% 
    layer_flatten()
  
  month_emb = month_input %>% 
    layer_embedding(13, 1) %>% 
    layer_flatten()
  
  year_emb = year_input %>% 
    layer_embedding(7, 1) %>% 
    layer_flatten()
  
  event_name_1_emb = event_name_1_input %>% 
    layer_embedding(32, 1) %>% 
    layer_flatten()
  
  event_type_1_emb = event_type_1_input %>% 
    layer_embedding(6, 1) %>% 
    layer_flatten()



  event_name_2_emb = event_name_2_input %>% 
    layer_embedding(6, 1) %>% 
    layer_flatten()
  
  event_type_2_emb = event_type_2_input %>% 
    layer_embedding(6, 1) %>% 
    layer_flatten()
  
  item_id_emb = item_id_input %>% 
    layer_embedding(3050, 3) %>% 
    layer_flatten()
  
  dept_id_emb = dept_id_input %>% 
    layer_embedding(8, 1) %>% 
    layer_flatten()
  
  store_id_emb = store_id_input %>% 
    layer_embedding(11, 1) %>% 
    layer_flatten()
  
  cat_id_emb = cat_id_input %>% 
    layer_embedding(4, 1) %>% 
    layer_flatten()
  
  state_id_emb = state_id_input %>% 
    layer_embedding(4, 1) %>% 
    layer_flatten()
  
  # Combine everyting
  outputs <- list(wday_emb, month_emb, year_emb, event_name_1_emb,
                  event_type_1_emb, event_name_2_emb, event_type_2_emb, 
                  item_id_emb, dept_id_emb, store_id_emb, cat_id_emb, state_id_emb) %>% 
    layer_concatenate() %>% 
    layer_dense(150, activation="relu") %>% 
    layer_dense(75, activation="relu") %>% 
    layer_dense(10, activation="relu") %>% 
    
    layer_dense(1, activation="linear", name = "output")
  
  inputs <- list(wday = wday_input, 
                 month = month_input,
                 year = year_input, 
                 event_name_1 = event_name_1_input, 
                 event_type_1 = event_type_1_input,
                 event_name_2 = event_name_2_input, 
                 event_type_2 = event_type_2_input,
                 item_id = item_id_input, 
                 dept_id = dept_id_input, 
                 store_id = store_id_input,
                 cat_id = cat_id_input,
                 state_id = state_id_input)
  
  model <- keras_model(inputs, outputs)
  
  model %>% 
    compile(loss = loss_mean_squared_error,
            metrics = "mse",
            optimizer = optimizer_rmsprop(lr = lr))
  
  return(model)
}

model <- new_model()

model %>% 
  summary()


# FIT MODEL
history <- model %>% 
  fit(X, y, 
      batch_size = 10000, 
      epochs = 600, 
      verbose = 2,
      validation_data = valid)


#save.image("M5.RData")
#load("M5.RData")
# Makes list of input blocks
make_X <- function(df) {
  c(dense1 = list(data.matrix(df[, dense_cols, with = FALSE])), 
    as.list(df[, cat_cols, with = FALSE]))
}



pred <- model %>% predict(test,batch_size=10000)

test[, demand := pmax(0, pred)]
test_long <- dcast(test, id ~ F, value.var = "demand")
submission <- merge(sample_submission[, .(id)], 
                    test_long[, colnames(sample_submission), with = FALSE], 
                    by = "id")
fwrite(submission, file = "submission_keras.csv", row.names = FALSE)












# library(keras)
# data_dim <- 28
# timesteps <- 30490
#num_classes <- 10
# 
# model <- keras_model_sequential() %>% 
#   layer_lstm(units = 64, return_sequences = TRUE, input_shape = list(1, data_dim) , activation = "relu") %>% 
#   layer_lstm(units = 32, return_sequences = TRUE, activation = "relu") %>% 
#   layer_lstm(units = 8 , activation = "relu") %>% # return a single vector dimension 8
#   layer_dense(units = 1 , activation = "relu")
#   
#  
# model %>% compile(
#   optimizer = optimizer_rmsprop(),
#   loss = "mae"
# )
# 
# #----------------Understood the timestep concept--------------------
# # history <- model %>% fit(
# #   reticulate::array_reshape(X,c(dim(X)[[1]]/timesteps,timesteps,dim(X)[[-1]])),y,
# #   epochs = 5,
# #   validation_data = valid
# # )
# 
# history <- model %>% fit(
#   reticulate::array_reshape(X,c(dim(X)[[1]],NULL,dim(X)[[-1]])),y,
#   epochs = 5,
#   validation_data = valid
# )



# 
# model <- keras_model_sequential() %>% 
#   layer_dense(units = 64, activation = 'relu', input_shape = c(28)) %>%
#   layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2) %>% 
#   layer_gru(units = 8, dropout = 0.2, recurrent_dropout = 0.2) %>% 
#   layer_dense(units = 1)
# 
# 
# model %>% compile(
#   optimizer = optimizer_rmsprop(),
#   loss = "mae"
# )
# 
# history <- model %>% fit(
#   list(X,y),
#   batch_size = 32,
#   epochs = 5,
#   validation_data = valid
# )
# #Generator functions --------------------------------
# generator <- function(data, lookback, delay, min_index, max_index,
#                       shuffle = FALSE, batch_size = 32, step = 7) {
#   if (is.null(max_index))
#     max_index <- nrow(data) - delay - 1
#   i <- min_index + lookback
#   function() {
#     if (shuffle) {
#       rows <- sample(c((min_index+lookback):max_index), size = batch_size)
#     } else {
#       if (i + batch_size >= max_index)
#         i <<- min_index + lookback
#       rows <- c(i:min(i+batch_size-1, max_index))
#       i <<- i + length(rows)
#     }
#     
#     samples <- array(0, dim = c(length(rows),
#                                 lookback / step,
#                                 dim(data)[[-1]]))
#     targets <- array(0, dim = c(length(rows)))
#     
#     for (j in 1:length(rows)) {
#       indices <- seq(rows[[j]] - lookback, rows[[j]]-1,
#                      length.out = dim(samples)[[2]])
#       samples[j,,] <- data[indices,]
#       targets[[j]] <- data[rows[[j]] + delay,2]
#     }           
#     list(samples, targets)
#   }
# }
# 
# lookback <- 90
# step <- 2
# delay <- 10
# batch_size <- 32
# 
# train_gen <- generator(
#   X,
#   lookback = lookback,
#   delay = delay,
#   min_index = 1,
#   max_index = dim(X)[[1]],
#   shuffle = F,
#   step = step, 
#   batch_size = batch_size
# )
# 
# val_gen = generator(
#   valid[[1]],
#   lookback = lookback,
#   delay = delay,
#   min_index = 1,
#   max_index = dim(valid[[1]])[[1]],
#   step = step,
#   batch_size = batch_size
# )
# 
# test_gen <- generator(
#   test,
#   lookback = lookback,
#   delay = delay,
#   min_index = 1,
#   max_index = dim(test)[[1]],
#   step = step,
#   batch_size = batch_size
# )
# 
# # How many steps to draw from val_gen in order to see the entire validation set
# val_steps <- as.integer((dim(valid[[1]])[[1]] - lookback) / batch_size)
# 
# # How many steps to draw from test_gen in order to see the entire test set
# test_steps <- as.integer((dim(test)[[1]] - lookback) / batch_size)
# 
# 














